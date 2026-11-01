#!/usr/bin/env python3
"""Step 2: population anatomy model (probabilistic atlas + deformation PCA).

Input: the per-subject label maps of anatomy.py (1 mm grid centred on the CTV).

1. Registration images: each label map is turned into a "cartoon" (every class set to
   its population median intensity), the field of view gap filled with the nearest
   label, and downsampled to 2 mm.
2. Groupwise template (ITERS rounds): every subject is registered to the current
   template (affine, then diffeomorphic demons); the template is the mean of the
   warped class one-hots, unbiased by the inverse of the mean displacement.
3. Atlas: class probabilities P_c(x) on the 1 mm grid = field-of-view-weighted mean of
   the subjects' one-hots warped into template space.
4. Deformation model: each subject's inverse field phi_s (subject grid -> template) is
   a sample of anatomical variation; PCA over subjects gives mean field and modes.
   A new anatomy is P(y + phi(y)) with phi = mean + sum_i z_i sqrt(lambda_i) e_i.

Outputs in --out: atlas.npz (population level: P, mean field, modes, eigenvalues, class
values) and private/ (per-subject fields and PCA coefficients, for the privacy check;
never released).
"""
import argparse, glob, json, os
from multiprocessing import Pool
import numpy as np
import nibabel as nib
import scipy.ndimage as ndi
import SimpleITK as sitk

import anatomy as A

NC = len(A.CLASSES)
DS = 2           # registration / field resolution (mm)
ITERS = 3
N_MODES = 30


def load(anat, sid):
    lab = np.asarray(nib.load(f"{anat}/{sid}_lab.nii.gz").dataobj)
    img = np.asarray(nib.load(f"{anat}/{sid}_img.nii.gz").dataobj, np.float32)
    return lab, img


def fill_fov(lab):
    """Nearest-label fill of voxels outside the field of view; returns (filled, fov mask)."""
    fov = lab != A.OUTSIDE_FOV
    if fov.all():
        return lab, fov
    idx = ndi.distance_transform_edt(~fov, return_distances=False, return_indices=True)
    return lab[tuple(idx)], fov


def to_sitk(arr_xyz, spacing):
    im = sitk.GetImageFromArray(np.ascontiguousarray(np.transpose(arr_xyz, (2, 1, 0))).astype(np.float32))
    im.SetSpacing([float(spacing)] * 3)
    im.SetOrigin([-(n - 1) / 2 * 1.0 for n in arr_xyz.shape])   # mm, grid centre at 0 (1 mm grid convention)
    return im


def from_sitk(im):
    a = sitk.GetArrayFromImage(im)
    return np.transpose(a, (2, 1, 0) + tuple(range(3, a.ndim)))


def cartoon(lab_filled, values, ds=DS):
    c = values[lab_filled].astype(np.float32)
    c = ndi.gaussian_filter(c, 1.0)
    im = to_sitk(c, 1.0)
    return sitk.Resample(im, [n // ds for n in c.shape], sitk.Transform(), sitk.sitkLinear,
                         im.GetOrigin(), [float(ds)] * 3)


def register(fixed, moving):
    """Dense displacement field on the fixed grid mapping fixed -> moving points."""
    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsMeanSquares()
    R.SetOptimizerAsRegularStepGradientDescent(2.0, 1e-3, 200, relaxationFactor=0.7)
    R.SetOptimizerScalesFromPhysicalShift()
    R.SetShrinkFactorsPerLevel([4, 2, 1])
    R.SetSmoothingSigmasPerLevel([2.0, 1.0, 0.0])
    R.SetInterpolator(sitk.sitkLinear)
    aff = sitk.AffineTransform(3)
    R.SetInitialTransform(aff, inPlace=True)
    R.Execute(fixed, moving)
    moved = sitk.Resample(moving, fixed, aff, sitk.sitkLinear, 0.0)
    disp = None
    for shrink, iters in ((2, 80), (1, 60)):
        f = sitk.Shrink(fixed, [shrink] * 3) if shrink > 1 else fixed
        m = sitk.Shrink(moved, [shrink] * 3) if shrink > 1 else moved
        d = sitk.FastSymmetricForcesDemonsRegistrationFilter()
        d.SetNumberOfIterations(iters)
        d.SetStandardDeviations(1.5)
        d.SetSmoothUpdateField(True); d.SetUpdateFieldStandardDeviations(1.0)
        if disp is None:
            disp = d.Execute(f, m)
        else:
            init = sitk.Resample(disp, f, sitk.Transform(), sitk.sitkLinear, 0.0, disp.GetPixelID())
            disp = d.Execute(f, m, init)
    disp = sitk.Resample(disp, fixed, sitk.Transform(), sitk.sitkLinear, 0.0, sitk.sitkVectorFloat64)
    comp = sitk.CompositeTransform(aff)
    comp.AddTransform(sitk.DisplacementFieldTransform(sitk.Image(disp)))
    return sitk.TransformToDisplacementField(comp, sitk.sitkVectorFloat64, fixed.GetSize(),
                                             fixed.GetOrigin(), fixed.GetSpacing(), fixed.GetDirection())


def warp_onehot(lab, fov, field, ref_shape):
    """Warp class one-hots (1 mm, subject space) into template space with a DS-mm field
    (template -> subject). Returns (onehot float32 (NC, *ref_shape), fov weight)."""
    u = from_sitk(field)                                    # (x/DS, y/DS, z/DS, 3) mm
    up = np.stack([ndi.zoom(u[..., k], np.array(ref_shape) / np.array(u.shape[:3]), order=1)
                   for k in range(3)], -1)
    g = np.stack(np.meshgrid(*[np.arange(n, dtype=np.float32) for n in ref_shape], indexing="ij"), -1)
    coords = (g + up).reshape(-1, 3).T                      # 1 mm voxel coords in subject grid
    w = ndi.map_coordinates(fov.astype(np.float32), coords, order=1, cval=0.0).reshape(ref_shape)
    oh = np.empty((NC,) + tuple(ref_shape), np.float32)
    for c in range(NC):
        oh[c] = ndi.map_coordinates((lab == c).astype(np.float32), coords, order=1, mode="nearest").reshape(ref_shape)
    return oh, w


def _reg_job(args):
    sid, anat, values, tmpl_path = args
    lab, _ = load(anat, sid)
    labf, fov = fill_fov(lab)
    fixed = sitk.ReadImage(tmpl_path)
    field = register(fixed, cartoon(labf, values))
    return sid, sitk.GetArrayFromImage(field)


def _accum_job(args):
    sids, anat, fields, shape, spacing_info = args
    S = np.zeros((NC,) + tuple(shape), np.float32); W = np.zeros(shape, np.float32)
    for sid in sids:
        lab, _ = load(anat, sid)
        labf, fov = fill_fov(lab)
        f = sitk.GetImageFromArray(fields[sid], isVector=True)
        f.SetSpacing(spacing_info[0]); f.SetOrigin(spacing_info[1])
        oh, w = warp_onehot(labf, fov, f, shape)
        S += oh * w; W += w
    return S, W


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anat", required=True)
    ap.add_argument("--subjects", required=True, help="file: subjects to use (after exclusions)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--nproc", type=int, default=16)
    a = ap.parse_args()
    os.makedirs(f"{a.out}/private", exist_ok=True)
    sids = [l.strip() for l in open(a.subjects) if l.strip()]
    shape = tuple(A.GRID_MM)

    # population median intensity per class (cartoon values)
    vals = {c: [] for c in range(NC)}
    for sid in sids:
        lab, img = load(a.anat, sid)
        for c in range(NC):
            m = lab == c
            if m.sum() > 50:
                vals[c].append(float(np.median(img[m])))
    values = np.array([np.median(vals[c]) if vals[c] else 0.0 for c in range(NC)] + [0.0] * (256 - NC), np.float32)
    values[0] = 0.0

    # initial template: subject whose cartoon is closest to the mean cartoon
    carts = {}
    for sid in sids:
        lab, _ = load(a.anat, sid)
        carts[sid] = sitk.GetArrayFromImage(cartoon(fill_fov(lab)[0], values))
    mean_c = np.mean(list(carts.values()), 0)
    init = min(sids, key=lambda s: float(((carts[s] - mean_c) ** 2).mean()))
    ref = cartoon(fill_fov(load(a.anat, init)[0])[0], values)
    tmpl = sitk.GetImageFromArray(mean_c); tmpl.CopyInformation(ref)
    del carts

    spacing_info = (tmpl.GetSpacing(), tmpl.GetOrigin())
    chunks = lambda L, n: [L[i::n] for i in range(n)]
    with Pool(a.nproc) as pool:
        for it in range(ITERS + 1):
            tp = f"{a.out}/private/template_{it}.nii.gz"
            sitk.WriteImage(tmpl, tp)
            fields = dict(pool.map(_reg_job, [(s, a.anat, values, tp) for s in sids]))
            print(f"iter {it}: registered {len(fields)}", flush=True)
            if it == ITERS:
                break
            res = pool.map(_accum_job, [(ch, a.anat, {s: fields[s] for s in ch}, shape, spacing_info)
                                        for ch in chunks(sids, a.nproc) if ch])
            S = sum(r[0] for r in res); W = sum(r[1] for r in res)
            P = S / np.maximum(W, 1e-3)
            cart1 = (P * values[:NC, None, None, None]).sum(0)
            new = sitk.Resample(to_sitk(cart1, 1.0), tmpl, sitk.Transform(), sitk.sitkLinear, 0.0)
            # unbias: warp by the inverse of the mean template->subject displacement
            ubar = np.mean([fields[s] for s in sids], 0)
            fb = sitk.GetImageFromArray(ubar, isVector=True); fb.CopyInformation(tmpl)
            inv = sitk.InvertDisplacementField(fb, 20, 0.1, 0.001, False)
            tmpl = sitk.Resample(new, sitk.DisplacementFieldTransform(inv))
            tmpl.CopyInformation(ref)
            print(f"iter {it}: mean |u| {np.linalg.norm(ubar, axis=-1).mean():.2f} mm", flush=True)

        # final atlas (1 mm) from the last registration round
        res = pool.map(_accum_job, [(ch, a.anat, {s: fields[s] for s in ch}, shape, spacing_info)
                                    for ch in chunks(sids, a.nproc) if ch])
    S = sum(r[0] for r in res); W = sum(r[1] for r in res)
    P = S / np.maximum(W, 1e-3)
    # regions seen by too few subjects: extend from the nearest well-covered voxel
    good = W >= 0.25 * len(sids)
    if not good.all():
        idx = ndi.distance_transform_edt(~good, return_distances=False, return_indices=True)
        P = P[(slice(None),) + tuple(idx)]

    # inverse fields (subject grid -> template) and PCA
    inv = {}
    for s in sids:
        f = sitk.GetImageFromArray(fields[s], isVector=True); f.CopyInformation(tmpl)
        inv[s] = sitk.GetArrayFromImage(sitk.InvertDisplacementField(f, 30, 0.1, 0.001, False)).astype(np.float32)
    X = np.stack([inv[s].ravel() for s in sids])
    mu = X.mean(0); Xc = X - mu
    Gm = Xc @ Xc.T
    ev, V = np.linalg.eigh(Gm)
    order = np.argsort(ev)[::-1][:min(N_MODES, len(sids) - 1)]
    ev, V = ev[order], V[:, order]
    modes = (Xc.T @ V) / np.sqrt(np.maximum(ev, 1e-9))     # unit-norm modes (columns)
    lam = ev / (len(sids) - 1)
    coeffs = (Xc @ modes) / np.sqrt(lam)                    # per-subject standardised coefficients
    np.savez_compressed(f"{a.out}/atlas.npz", P=(np.clip(P, 0, 1) * 255).astype(np.uint8),
                        classes=np.array(A.CLASSES), values=values[:NC], field_shape=np.array(inv[sids[0]].shape),
                        field_spacing=DS, mean=mu.astype(np.float32), modes=modes.T.astype(np.float32),
                        lam=lam.astype(np.float32), n_subjects=len(sids), grid_mm=np.array(shape))
    np.savez_compressed(f"{a.out}/private/subject_coeffs.npz", sids=np.array(sids), coeffs=coeffs.astype(np.float32))
    expl = np.cumsum(lam) / (np.trace(Gm) / (len(sids) - 1))
    json.dump(dict(subjects=sids, n=len(sids), explained_cumulative=expl.tolist()[:N_MODES],
                   values=dict(zip(A.CLASSES, values[:NC].tolist()))), open(f"{a.out}/atlas.json", "w"), indent=1)
    print("explained variance (cumulative):", np.round(expl[:10], 3))


if __name__ == "__main__":
    main()
