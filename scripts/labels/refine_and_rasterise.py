#!/usr/bin/env python3
"""Steps 2-4 of label correction.

2. Void map. For each T1 volume I (N4-corrected):
       I_s = Gaussian(I, sigma = 0.5, 0.5, 0.25 voxels)
       B   = median filter of I over 9 x 9 x 3 voxels (~6 x 6 x 6 mm, local tissue level)
       V   = clip((B - I_s) / B, 0, 1)            (fractional signal loss)
   Vd = V max-filtered over a sphere of radius TOL_MM (tolerance for small
   inter-marker deformation).

3. Rigid constellation refinement. The CT marker constellation p_i (already in MR
   space through the clinical registration) is rotated about its centroid and
   translated, (R, t) in a bounded grid (rotations +/-ROT_DEG about each axis,
   translations +/-TRANS_MM), maximising S(R, t) = sum_i Vd(R (p_i - c) + c + t).
   The refined transform is accepted only if it increases S by more than
   MIN_GAIN; otherwise the clinical registration is kept. Each marker is then
   snapped to the V-weighted centroid of the strongest void within SNAP_MM of its
   refined position if that void exceeds VIS_THR ("visible"); otherwise the
   constellation position is kept and the marker is flagged not visible.

4. Rasterisation on the MR grid. Markers: spheres of radius SPHERE_MM at the final
   positions (class 1). Prostate: the CTV contours from the RTSTRUCT, mapped with
   the clinical registration followed by the accepted rigid refinement, rasterised
   on the MR grid (class 2). Markers overwrite the prostate.

Outputs per subject: labels/<variant>/{sid}.nii.gz and a JSON record.
"""
import argparse, glob, itertools, json, os
import numpy as np
import nibabel as nib
import pandas as pd
import pydicom
import scipy.ndimage as ndi
from scipy.spatial.transform import Rotation
from matplotlib.path import Path

TOL_MM, ROT_DEG, ROT_STEP, TRANS_MM = 1.5, 6.0, 3.0, 6.0
MIN_GAIN, SNAP_MM, VIS_THR, SPHERE_MM = 0.4, 2.0, 0.45, 3.0
CTV_PRIORITY = ['CTV_High', 'CTV_Low', 'CTVp']


def void_map(vol):
    I_s = ndi.gaussian_filter(vol, (0.5, 0.5, 0.25))
    B = ndi.median_filter(vol, size=(9, 9, 3))
    return np.clip((B - I_s) / np.maximum(B, 1e-6), 0, 1)


def ball(r_mm, z):
    r = np.ceil(r_mm / z).astype(int)
    g = np.mgrid[-r[0]:r[0] + 1, -r[1]:r[1] + 1, -r[2]:r[2] + 1]
    return (((g * z[:, None, None, None]) ** 2).sum(0) <= r_mm ** 2)


def interp(vol, pts_vox):
    return ndi.map_coordinates(vol, pts_vox.T, order=1, mode='constant', cval=0.0)


def lps_to_vox(p_lps, inv_aff):
    p = np.atleast_2d(p_lps) * np.array([-1, -1, 1])
    return (inv_aff @ np.c_[p, np.ones(len(p))].T).T[:, :3]


def refine(V, Vd, P_vox, z):
    """P_vox: markers in voxel coords. Returns refined voxel coords, params, scores."""
    P = P_vox * z  # mm in voxel-aligned frame
    c = P.mean(0)
    angles = np.arange(-ROT_DEG, ROT_DEG + 1e-6, ROT_STEP)
    rots = [Rotation.from_euler('xyz', a, degrees=True).as_matrix() for a in itertools.product(angles, angles, angles)]
    tx = np.arange(-TRANS_MM, TRANS_MM + 1e-6, z[0])
    tz = np.arange(-TRANS_MM, TRANS_MM + 1e-6, z[2] / 2)
    T = np.array(list(itertools.product(tx, tx, tz)))
    s0 = interp(Vd, P_vox).sum()
    best = (s0, np.eye(3), np.zeros(3))
    for Rm in rots:
        Q = (P - c) @ Rm.T + c                       # (n,3)
        pts = (Q[None, :, :] + T[:, None, :]) / z    # (nT,n,3) voxel
        s = interp(Vd, pts.reshape(-1, 3)).reshape(len(T), -1).sum(1)
        k = int(np.argmax(s))
        if s[k] > best[0]:
            best = (float(s[k]), Rm, T[k])
    accepted = best[0] - s0 > MIN_GAIN
    Rm, t = (best[1], best[2]) if accepted else (np.eye(3), np.zeros(3))
    Q = ((P - c) @ Rm.T + c + t) / z
    # snap
    out, vis = [], []
    rb = ball(SNAP_MM, z); rr = np.array(rb.shape) // 2
    for q in Q:
        qi = np.round(q).astype(int)
        lo = qi - rr; hi = qi + rr + 1
        if np.any(lo < 0) or np.any(hi > V.shape):
            out.append(q); vis.append(0.0); continue
        sub = np.where(rb, V[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]], 0)
        j = np.unravel_index(np.argmax(sub), sub.shape)
        vmax = float(sub[j])
        if vmax >= VIS_THR:
            # V-weighted centroid of voxels > 0.5 vmax connected to the peak
            m, _ = ndi.label(sub > 0.5 * vmax)
            m = m == m[j]
            w = sub * m
            g = np.indices(sub.shape).reshape(3, -1)
            cen = (g * w.ravel()).sum(1) / w.sum()
            out.append(lo + cen)
        else:
            out.append(q)
        vis.append(vmax)
    R_vox = np.array(out)
    return R_vox, dict(accepted=bool(accepted), gain=float(best[0] - s0), score_clinical=float(s0),
                       score_refined=float(best[0]), rot_deg=Rotation.from_matrix(Rm).as_euler('xyz', degrees=True).tolist(),
                       trans_mm=list(map(float, t)), centre_vox=list(map(float, c / z))), vis, Rm, t, c


def rasterise_ctv(rs, roi_name, reg_inv, aff, shape, z, Rm, t, c):
    """Fill CTV planar contours (CT frame) on a fine in-plane grid, map points to MR
    voxels (clinical registration then refinement), mark voxels, close holes."""
    num = next(r.ROINumber for r in rs.StructureSetROISequence if r.ROIName == roi_name)
    cs = next(c_ for c_ in rs.ROIContourSequence if c_.ReferencedROINumber == num)
    inv_aff = np.linalg.inv(aff)
    mask = np.zeros(shape, bool)
    zs = sorted({round(float(np.array(cc.ContourData, float)[2]), 2) for cc in cs.ContourSequence})
    dz = np.median(np.diff(zs)) if len(zs) > 1 else 2.0
    for cc in cs.ContourSequence:
        p = np.array(cc.ContourData, float).reshape(-1, 3)
        if len(p) < 3:
            continue
        x0, y0 = p[:, :2].min(0); x1, y1 = p[:, :2].max(0)
        gx, gy = np.meshgrid(np.arange(x0, x1 + 0.3, 0.3), np.arange(y0, y1 + 0.3, 0.3))
        inside = Path(p[:, :2]).contains_points(np.c_[gx.ravel(), gy.ravel()])
        xy = np.c_[gx.ravel(), gy.ravel()][inside]
        for off in np.arange(-dz / 2, dz / 2 + 1e-6, 0.5):
            pts = np.c_[xy, np.full(len(xy), p[0, 2] + off)]
            mr = (reg_inv @ np.c_[pts, np.ones(len(pts))].T).T[:, :3]
            v = lps_to_vox(mr, inv_aff)
            v = ((v * z - c) @ Rm.T + c + t) / z
            vi = np.round(v).astype(int)
            ok = np.all((vi >= 0) & (vi < shape), 1)
            mask[tuple(vi[ok].T)] = True
    mask = ndi.binary_closing(mask, np.ones((3, 3, 1)), iterations=1)
    for k in range(shape[2]):
        mask[:, :, k] = ndi.binary_fill_holes(mask[:, :, k])
    return mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--subjects', nargs='*')
    a = ap.parse_args()
    R = a.root
    outdir = os.path.join(R, 'labels', 'corrected'); os.makedirs(outdir, exist_ok=True)
    recdir = os.path.join(R, 'labelfix', 'refined'); os.makedirs(recdir, exist_ok=True)
    subs = a.subjects or [l.strip() for l in open(os.path.join(R, 'subjects.txt'))]
    rows = []
    for sid in subs:
        img = nib.load(os.path.join(R, 'images', f'{sid}.nii'))
        vol = np.asarray(img.dataobj, np.float32); aff = img.affine; inv_aff = np.linalg.inv(aff)
        z = np.array(img.header.get_zooms()[:3], float)
        mrec_p = os.path.join(R, 'labelfix', 'markers', f'{sid}.json')
        mrec = json.load(open(mrec_p)) if os.path.exists(mrec_p) else None
        old = np.asarray(nib.load(os.path.join(R, 'labels', 'original', f'{sid}.nii.gz')).dataobj).astype(np.int16)
        if mrec and mrec.get('markers'):
            P = lps_to_vox(np.array([m['mr_lps'] for m in mrec['markers']]), inv_aff)
            init = 'dicom'
        else:
            l, n = ndi.label(old == 1, np.ones((3, 3, 3)))
            P = np.array([ndi.center_of_mass(l == i) for i in range(1, n + 1)])
            init = 'existing_label'
        V = void_map(vol)
        Vd = ndi.maximum_filter(V, footprint=ball(TOL_MM, z))
        Rv, info, vis, Rm, t, c = refine(V, Vd, P, z)
        # prostate
        ctv_note = 'rtstruct'
        try:
            if init != 'dicom' or not mrec.get('ctv_roi'):
                raise ValueError('no rtstruct ctv')
            d = glob.glob(os.path.join(R, 'dicoms', mrec['archive'], 'RS', mrec['rtstruct']))[0]
            rs = pydicom.dcmread(d)
            reg_inv = np.linalg.inv(np.array(mrec['reg_matrix']))
            ctv = rasterise_ctv(rs, mrec['ctv_roi'], reg_inv, aff, vol.shape, z, Rm, t, c)
        except Exception as e:
            ctv = old >= 1
            ctv_note = f'existing_label ({e})'
        lab = np.zeros(vol.shape, np.uint8)
        lab[ctv] = 2
        g = np.indices(vol.shape).reshape(3, -1).T
        for q in Rv:
            lo = np.maximum(np.floor(q - SPHERE_MM / z).astype(int), 0); hi = np.minimum(np.ceil(q + SPHERE_MM / z).astype(int) + 1, vol.shape)
            sl = tuple(slice(l_, h_) for l_, h_ in zip(lo, hi))
            gg = np.indices(hi - lo).reshape(3, -1).T + lo
            inside = (((gg - q) * z) ** 2).sum(1) <= SPHERE_MM ** 2
            sub = lab[sl].reshape(-1); sub[inside] = 1; lab[sl] = sub.reshape(hi - lo)
        nib.save(nib.Nifti1Image(lab, aff), os.path.join(outdir, f'{sid}.nii.gz'))
        move = np.linalg.norm((Rv - P) * z, axis=1)
        old_ctv = old >= 1
        dice = 2 * (ctv & old_ctv).sum() / max(ctv.sum() + old_ctv.sum(), 1)
        rec = dict(subject=sid, init=init, status=mrec.get('status') if mrec else 'no_record', **info,
                   markers_init_vox=P.tolist(), markers_final_vox=Rv.tolist(), void_strength=vis,
                   move_mm=move.tolist(), ctv=ctv_note, ctv_dice_vs_original=float(dice))
        json.dump(rec, open(os.path.join(recdir, f'{sid}.json'), 'w'), indent=1)
        rows.append(dict(subject=sid, init=init, status=rec['status'], n=len(Rv), accepted=info['accepted'],
                         gain=round(info['gain'], 2), trans_mm=round(float(np.linalg.norm(info['trans_mm'])), 2),
                         rot_max=round(float(np.max(np.abs(info['rot_deg']))), 1), max_move_mm=round(float(move.max()), 2),
                         n_visible=int(sum(v >= VIS_THR for v in vis)), min_void=round(float(min(vis)), 2),
                         ctv=ctv_note.split(' ')[0], ctv_dice_vs_original=round(float(dice), 3)))
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(R, 'labelfix', f'refine_summary{"_partial" if a.subjects else ""}.csv'), index=False)


if __name__ == '__main__':
    main()
