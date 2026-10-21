#!/usr/bin/env python3
"""Model-free reference marker labels from CT (or RTSTRUCT points) without a clinical registration.

For patients whose archive has no usable CT->MR registration (or no RTSTRUCT at all),
marker positions on the T1 are derived automatically, independent of any network:

1. Source points (patient LPS mm, source frame), by priority:
   * CT: gold markers saturate the CT (about 3070 HU). Connected components of
     HU > CT_THR are split at HU maxima >= SPLIT_MM apart (touching markers); parts
     with volume VOL_MIN..VOL_MAX mm^3, peak >= PEAK_MIN HU and at least MIN_DEPTH_MM
     inside the body (skin, couch and index-bar markers excluded) are clustered (single linkage, LINK_MM). The largest cluster of 2..MAX_MARKERS
     components is the marker set (ties: highest summed peak HU). The marker
     centroid is the (HU - CT_THR)-weighted centroid of the component.
   * otherwise RTSTRUCT marker ROIs in any frame (derive_markers.py naming rules).
2. MR void map V of the N4-corrected T1 (refine_and_rasterise.void_map), its
   1.5 mm max-filtered version Vd, and void blobs: connected components of
   V >= BLOB_V of BLOB_VOL mm^3 within ROI_MM in-plane of the image centre
   (reference markers in the study set lie within 36 mm), at their V-weighted centroid.
3. Rigid correspondence search. Inter-marker distances are invariant to rigid
   motion. Every source triple is matched to MR blob triples whose three
   distances agree within PAIR_TOL mm (Kabsch fit). If no triple fits (a marker
   without a blob, or 2-marker sets), for every source pair (a, b) and MR blob pair (p, q) whose
   distance agrees within PAIR_TOL mm, the rotation taking b-a to q-p is
   completed by a spin about that axis (SPIN_STEP grid); hypotheses with a
   total rotation above MAX_ROT_DEG are dropped (correct fits on the tuning half:
   at most 12.5 deg), as are those that shift the marker
   centroid left-right by more than MAX_DX_MM (the patient is centred left-right
   on both scanners; study set: within 14 mm), or that put the marker centroid more
   than CEN_R_MM in-plane / CEN_Z_MM through-plane from the MR image centre (the
   T1 field of view is centred on the prostate). Each hypothesis is scored by the summed Vd
   at all transformed source points (so a marker with a weak void still counts
   through the others). The best hypotheses are polished by alternating snap to
   the strongest void within SNAP_MM and Kabsch refit.
4. The winning transform places every source marker on the MR grid, then the
   existing refinement (refine_and_rasterise.refine: small rigid search +
   per-marker void snap) gives the final positions.

Confidence: rms residual of the final Kabsch fit (source vs final MR positions),
the number of markers on a clear void (>= 0.45), and the margin between the best
and the best *different* hypothesis (one whose placements differ by > 3 mm).
The acceptance rule is set on a tuning half of the study subjects and reported
on the other half (see --validate).

T1 preparation for new patients (prepare): DICOM series -> NIfTI in the voxel layout
dcm2niix gave the study images (rows stored bottom-up), then the same N4 correction
(raw/scripts/preprocessing/homogeneity_correction.py). On a study subject this
reproduces the raw NIfTI exactly and images/{sid}.nii to 1e-5 relative.

Usage:
  auto_ct_labels.py prepare --series-dir <dir with the T1 seed DICOMs> [--uid <SeriesInstanceUID>] --out <sid>.nii
  auto_ct_labels.py validate --out <dir>          # study subjects with CT, blind to labels
  auto_ct_labels.py apply --dicom <archive dir> --t1 <N4 T1 .nii> --sid <id> --out <dir>
"""
import argparse, glob, itertools, json, os, re, sys
import numpy as np
import nibabel as nib
import pydicom
import scipy.ndimage as ndi
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from refine_and_rasterise import void_map, ball, interp, refine, TOL_MM, VIS_THR  # noqa: E402
from derive_markers import is_marker, roi_centroid  # noqa: E402

P = '/scratch/user/uqaste15/prostate-v2'
CT_THR, PEAK_MIN, VOL_MIN, VOL_MAX, LINK_MM, MAX_MARKERS, SPLIT_MM = 2000.0, 2500.0, 4.0, 80.0, 60.0, 5, 5.0
MIN_DEPTH_MM = 30.0  # gold markers lie deep in the pelvis; skin/couch markers do not
PAIR_TOL, MAX_ROT_DEG, SPIN_STEP, SNAP_MM = 2.5, 13.0, 2.0, 2.0
BLOB_V, BLOB_VOL, ROI_MM = 0.45, (2.0, 60.0), 55.0
MAX_DX_MM = 20.0  # left-right shift of the marker centroid, CT -> MR (study set: within 14 mm)
CEN_R_MM, CEN_Z_MM = 30.0, 25.0  # MR FoV is centred on the prostate (tuning half: centroid within 23.5 mm in-plane, 16.4 mm in z)


# ---------------------------------------------------------------- source points
def load_ct(d):
    ds = []
    for f in glob.glob(os.path.join(d, '**', '*'), recursive=True):
        if not os.path.isfile(f):
            continue
        try:
            x = pydicom.dcmread(f, stop_before_pixels=True)
        except Exception:
            continue
        if getattr(x, 'Modality', '') == 'CT' and hasattr(x, 'ImagePositionPatient'):
            ds.append(f)
    if not ds:
        return None
    hdr = [pydicom.dcmread(f, stop_before_pixels=True) for f in ds]
    ser = {}
    for f, h in zip(ds, hdr):
        ser.setdefault(h.SeriesInstanceUID, []).append(f)
    files = max(ser.values(), key=len)
    xs = [pydicom.dcmread(f) for f in files]
    iop = np.array(xs[0].ImageOrientationPatient, float); n = np.cross(iop[:3], iop[3:])
    xs.sort(key=lambda x: np.dot(n, np.array(x.ImagePositionPatient, float)))
    vol = np.stack([x.pixel_array * float(x.RescaleSlope) + float(x.RescaleIntercept) for x in xs], -1)
    ps = np.array(xs[0].PixelSpacing, float); pos = np.array([x.ImagePositionPatient for x in xs], float)
    A = np.eye(4); A[:3, 0] = iop[3:] * ps[0]; A[:3, 1] = iop[:3] * ps[1]
    A[:3, 2] = (pos[-1] - pos[0]) / max(len(pos) - 1, 1); A[:3, 3] = pos[0]
    return vol, A, dict(ct_series=xs[0].SeriesInstanceUID, ct_for=getattr(xs[0], 'FrameOfReferenceUID', ''),
                        ct_slices=len(xs), ct_zrange=[float(pos[0, 2]), float(pos[-1, 2])])


def body_depth(vol, A):
    """Distance (mm) to the body surface: largest component of HU > -400, holes filled per slice."""
    lab, n = ndi.label(vol > -400)
    if n == 0:
        return np.zeros(vol.shape, np.float32)
    body = lab == (np.argmax(ndi.sum(np.ones_like(lab), lab, range(1, n + 1))) + 1)
    for k in range(body.shape[2]):
        body[..., k] = ndi.binary_fill_holes(body[..., k])
    return ndi.distance_transform_edt(body, sampling=np.linalg.norm(A[:3, :3], axis=0)).astype(np.float32)


def ct_markers(vol, A):
    vox_mm3 = abs(np.linalg.det(A[:3, :3]))
    depth = body_depth(vol, A)
    lab, n = ndi.label(vol > CT_THR, np.ones((3, 3, 3)))
    comps = []
    for i, sl in enumerate(ndi.find_objects(lab), 1):
        m = lab[sl] == i
        if m.sum() * vox_mm3 > 4 * VOL_MAX:   # bone / prosthesis
            continue
        idx = np.argwhere(m) + [s.start for s in sl]; v = vol[sl][m]
        c = (A @ np.c_[idx, np.ones(len(idx))].T).T[:, :3]
        # split touching markers: HU maxima >= SPLIT_MM apart, voxels go to the nearest maximum
        mx = []
        for j in np.argsort(-v):
            if all(np.linalg.norm(c[j] - c[k]) >= SPLIT_MM for k in mx):
                if v[j] >= CT_THR + 100:
                    mx.append(j)
        if not mx:
            continue
        near = np.argmin(np.linalg.norm(c[:, None] - c[mx][None], axis=2), axis=1)
        for k in range(len(mx)):
            sel = near == k; vm = sel.sum() * vox_mm3; vk = v[sel]
            if not (VOL_MIN <= vm <= VOL_MAX and vk.max() >= PEAK_MIN):
                continue
            ii = idx[sel][np.argmax(vk)]
            if depth[tuple(ii)] < MIN_DEPTH_MM:
                continue
            w = vk - CT_THR
            comps.append(dict(lps=(c[sel] * w[:, None]).sum(0) / w.sum(), vol_mm3=float(vm), peak=float(vk.max())))
    if not comps:
        return [], dict(n_components=0)
    X = np.array([c['lps'] for c in comps])
    D = np.linalg.norm(X[:, None] - X[None], axis=2)
    # single-linkage clusters via union-find
    par = list(range(len(comps)))
    def f(i):
        while par[i] != i:
            par[i] = par[par[i]]; i = par[i]
        return i
    for i, j in zip(*np.where(D < LINK_MM)):
        par[f(i)] = f(j)
    groups = {}
    for i in range(len(comps)):
        groups.setdefault(f(i), []).append(i)
    ok = [g for g in groups.values() if 2 <= len(g) <= MAX_MARKERS]
    info = dict(n_components=len(comps), n_clusters=len(groups), cluster_sizes=sorted(len(g) for g in groups.values()))
    if not ok:
        return [], info
    # prefer clusters of saturated (gold) components, then size, then summed peak
    ok.sort(key=lambda g: (sum(comps[i]['peak'] >= 2800 for i in g), len(g), sum(comps[i]['peak'] for i in g)), reverse=True)
    info['ambiguous_cluster'] = len(ok) > 1 and len(ok[1]) == len(ok[0])
    return [comps[i] for i in ok[0]], info


def rs_markers(d):
    """Marker ROI centroids from RTSTRUCTs (any frame), latest structure set with marker ROIs."""
    cands = []
    for f in glob.glob(os.path.join(d, '**', '*'), recursive=True):
        if not os.path.isfile(f):
            continue
        try:
            rs = pydicom.dcmread(f)
        except Exception:
            continue
        if getattr(rs, 'Modality', '') != 'RTSTRUCT':
            continue
        names = {r.ROINumber: r.ROIName for r in rs.StructureSetROISequence}
        if any(is_marker(nm) for nm in names.values()):
            cands.append((str(getattr(rs, 'StructureSetDate', '')) + str(getattr(rs, 'StructureSetTime', '')), f, rs, names))
    if not cands:
        return [], {}
    cands.sort(key=lambda c: c[0])
    _, f, rs, names = cands[-1]
    contours = {c.ReferencedROINumber: c for c in getattr(rs, 'ROIContourSequence', [])}
    pts = []
    for num, nm in names.items():
        if is_marker(nm) and num in contours and hasattr(contours[num], 'ContourSequence'):
            c, _ = roi_centroid(contours[num].ContourSequence)
            if c is not None:
                for q in pts:
                    if np.linalg.norm(np.mean(q['pts'], 0) - c) < 4.0:
                        q['pts'].append(c); break
                else:
                    pts.append(dict(pts=[c]))
    return [dict(lps=np.mean(q['pts'], 0)) for q in pts], dict(rtstruct=os.path.basename(f),
                                                                 rs_for=rs.ReferencedFrameOfReferenceSequence[0].FrameOfReferenceUID)


# ---------------------------------------------------------------- MR side
def mr_maps(t1_path, cache=None):
    img = nib.load(t1_path); z = np.array(img.header.get_zooms()[:3], float)
    if cache and os.path.exists(cache):
        V = np.load(cache)['V']
    else:
        V = void_map(np.asarray(img.dataobj, np.float32))
    Vd = ndi.maximum_filter(V, footprint=ball(TOL_MM, z))
    sh = np.array(V.shape)
    off = (np.indices(sh) - ((sh - 1) / 2)[:, None, None, None]) * z[:, None, None, None]
    roi = (np.abs(off[0]) <= ROI_MM) & (np.abs(off[1]) <= ROI_MM)
    lab, n = ndi.label((V >= BLOB_V) & roi)
    idx = np.arange(1, n + 1)
    vol = ndi.sum(np.ones_like(V), lab, idx) * np.prod(z)
    cen = np.array(ndi.center_of_mass(V, lab, idx)).reshape(-1, 3)
    keep = (vol >= BLOB_VOL[0]) & (vol <= BLOB_VOL[1])
    return img, z, V, Vd, cen[keep], vol[keep]


def lps_of_vox(A, v):
    return (A @ np.c_[v, np.ones(len(v))].T).T[:, :3] * [-1, -1, 1]


def vox_of_lps(A, p):
    return (np.linalg.inv(A) @ np.c_[np.atleast_2d(p) * [-1, -1, 1], np.ones(len(np.atleast_2d(p)))].T).T[:, :3]


def kabsch(a, b):
    ca, cb = a.mean(0), b.mean(0)
    U, S, Vt = np.linalg.svd((a - ca).T @ (b - cb))
    D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    return R, cb - ca @ R.T


def rot_between(u, v):
    u = u / np.linalg.norm(u); v = v / np.linalg.norm(v)
    ax = np.cross(u, v); s = np.linalg.norm(ax); c = np.dot(u, v)
    if s < 1e-9:
        return np.eye(3) if c > 0 else Rotation.from_rotvec(np.pi * np.array([1, 0, 0])).as_matrix()
    return Rotation.from_rotvec(ax / s * np.arctan2(s, c)).as_matrix()


def snap_points(V, z, Q_vox):
    """Move each point to the V-argmax within SNAP_MM (no threshold); returns positions and V there."""
    rb = ball(SNAP_MM, z); rr = np.array(rb.shape) // 2; out, val = [], []
    for q in Q_vox:
        qi = np.round(q).astype(int); lo = qi - rr; hi = qi + rr + 1
        if np.any(lo < 0) or np.any(hi > V.shape):
            out.append(q); val.append(0.0); continue
        sub = np.where(rb, V[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]], 0)
        j = np.unravel_index(np.argmax(sub), sub.shape)
        out.append(lo + np.array(j)); val.append(float(sub[j]))
    return np.array(out, float), np.array(val)


def match(src, img, z, V, Vd, peaks, top=20):
    A = img.affine
    S = np.array(src)  # source LPS
    Pk = lps_of_vox(A, peaks)
    DP = np.linalg.norm(Pk[:, None] - Pk[None], axis=2)
    hyps = []
    if len(S) >= 3:
        for tri in itertools.combinations(range(len(S)), 3):
            a, b, c = tri; d = lambda i, j: np.linalg.norm(S[i] - S[j])
            ii, jj = np.where(np.abs(DP - d(a, b)) < PAIR_TOL)
            for p, q in zip(ii, jj):
                ks = np.where((np.abs(DP[p] - d(a, c)) < PAIR_TOL) & (np.abs(DP[q] - d(b, c)) < PAIR_TOL))[0]
                for k in ks:
                    R, t = kabsch(S[[a, b, c]], Pk[[p, q, k]])
                    if np.degrees(Rotation.from_matrix(R).magnitude()) <= MAX_ROT_DEG:
                        hyps.append((R, t))
    cS = S.mean(0); sh = np.array(V.shape)

    def plausible(h):
        c = cS @ h[0].T + h[1]
        if abs((c - cS)[0]) > MAX_DX_MM:
            return False
        o = (vox_of_lps(A, c)[0] - (sh - 1) / 2) * z
        return np.hypot(o[0], o[1]) <= CEN_R_MM and abs(o[2]) <= CEN_Z_MM
    hyps = [h for h in hyps if plausible(h)]
    triple_hyps = len(hyps)
    spins = np.radians(np.arange(-MAX_ROT_DEG, MAX_ROT_DEG + 1e-6, SPIN_STEP))
    for a, b in (itertools.permutations(range(len(S)), 2) if not hyps else []):
        dab = np.linalg.norm(S[b] - S[a])
        ii, jj = np.where(np.abs(DP - dab) < PAIR_TOL)
        if not len(ii):
            continue
        u = S[b] - S[a]; un = u / np.linalg.norm(u)
        for p, q in zip(ii, jj):
            R0 = rot_between(u, Pk[q] - Pk[p]); wn = (Pk[q] - Pk[p]) / np.linalg.norm(Pk[q] - Pk[p])
            for th in spins:
                R = Rotation.from_rotvec(wn * th).as_matrix() @ R0
                if np.degrees(Rotation.from_matrix(R).magnitude()) > MAX_ROT_DEG:
                    continue
                t = Pk[p] - S[a] @ R.T
                hyps.append((R, t))
    hyps = [h for h in hyps if plausible(h)]
    if not hyps:
        return None
    Rs = np.array([h[0] for h in hyps]); ts = np.array([h[1] for h in hyps])
    Q = np.einsum('hij,nj->hni', Rs, S) + ts[:, None, :]           # (H, n, 3) LPS
    Qv = vox_of_lps(A, Q.reshape(-1, 3)).reshape(Q.shape)
    sc = interp(Vd, Qv.reshape(-1, 3)).reshape(len(hyps), -1).sum(1)
    order = np.argsort(-sc)
    res, seen = [], []
    for h in order:
        if len(res) >= top:
            break
        if any(np.abs(Q[h] - Q[s]).max() < 1.0 for s in seen):
            continue
        seen.append(h)
        R, t = Rs[h], ts[h]
        for _ in range(4):  # polish: snap + Kabsch
            Qh = S @ R.T + t
            sv, vals = snap_points(V, z, vox_of_lps(A, Qh))
            good = vals >= VIS_THR
            if good.sum() >= 3 or (good.sum() == 2 and len(S) == 2):
                R2, t2 = kabsch(S[good], lps_of_vox(A, sv[good]))
                if np.degrees(Rotation.from_matrix(R2).magnitude()) <= MAX_ROT_DEG:
                    R, t = R2, t2
            elif good.sum() == 2:
                t = t + (lps_of_vox(A, sv[good]) - Qh[good]).mean(0)
        Qh = S @ R.T + t
        sc2 = float(interp(Vd, vox_of_lps(A, Qh)).sum())
        res.append(dict(R=R, t=t, score=sc2, Q=Qh))
    res.sort(key=lambda r: -r['score'])
    best = res[0]
    other = [r for r in res[1:] if np.linalg.norm(r['Q'] - best['Q'], axis=1).max() > 3.0]
    best['margin'] = best['score'] - (other[0]['score'] if other else 0.0)
    best['n_hyp'] = len(hyps); best['triple_hyps'] = triple_hyps
    return best


def label_subject(src_pts, t1_path, cache=None):
    img, z, V, Vd, peaks, pv = mr_maps(t1_path, cache)
    S = np.array([s['lps'] for s in src_pts])
    m = match(S, img, z, V, Vd, peaks)
    if m is None:
        return None
    P_vox = vox_of_lps(img.affine, m['Q'])
    R_vox, rinfo, vis, *_ = refine(V, Vd, P_vox, z)
    fin = lps_of_vox(img.affine, R_vox)
    R, t = kabsch(S, fin)
    rms = float(np.sqrt(((S @ R.T + t - fin) ** 2).sum(1).mean()))
    return dict(markers_final_vox=R_vox.tolist(), markers_placed_vox=P_vox.tolist(), void_strength=list(map(float, vis)),
                n_clear_void=int(sum(v >= VIS_THR for v in vis)), rms_mm=rms,
                rot_deg=float(np.degrees(Rotation.from_matrix(R).magnitude())), score=m['score'], margin=float(m['margin']),
                n_hyp=m['n_hyp'], triple_hyps=m['triple_hyps'], n_mr_blobs=int(len(peaks)), refine=rinfo)


def qc_png(rec, t1_path, out_png, ct=None):
    """Row 1: CT MIP around the source markers (if CT). Row 2: T1 axial slice through each
    final marker (40 mm patch, final position red, placed-before-snap position cyan)."""
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    img = nib.load(t1_path); T = np.asarray(img.dataobj, np.float32); z = np.array(img.header.get_zooms()[:3])
    fin = np.array(rec.get('markers_final_vox', [])); pl = np.array(rec.get('markers_placed_vox', fin))
    n = max(len(fin), 1)
    fig, ax = plt.subplots(2, n + 1, figsize=(3 * (n + 1), 6))
    for a_ in ax.ravel():
        a_.axis('off')
    if ct is not None and rec.get('src_lps'):
        vol, Ac = ct; S = np.array(rec['src_lps']); iv = (np.linalg.inv(Ac) @ np.c_[S, np.ones(len(S))].T).T[:, :3]
        c = np.round(iv.mean(0)).astype(int); r = 50
        sub = vol[max(c[0] - r, 0):c[0] + r, max(c[1] - r, 0):c[1] + r, max(c[2] - 20, 0):c[2] + 20].max(2)
        ax[0, 0].imshow(sub, cmap='gray', vmin=-200, vmax=2000)
        ax[0, 0].plot(iv[:, 1] - max(c[1] - r, 0), iv[:, 0] - max(c[0] - r, 0), 'r+', ms=12)
        ax[0, 0].set_title('CT MIP (detected markers +)', fontsize=8)
    mid = np.round(fin.mean(0)).astype(int) if len(fin) else np.array(T.shape) // 2
    ax[1, 0].imshow(T[:, :, mid[2]].T, cmap='gray', origin='lower'); ax[1, 0].plot(fin[:, 0], fin[:, 1], 'r+', ms=8) if len(fin) else None
    ax[1, 0].set_title(f"T1 slice {mid[2]}", fontsize=8)
    h = int(round(20 / z[0]))
    for i, q in enumerate(fin):
        qi = np.round(q).astype(int); k = int(np.clip(qi[2], 0, T.shape[2] - 1))
        lo0, lo1 = max(qi[0] - h, 0), max(qi[1] - h, 0)
        patch = T[lo0:qi[0] + h, lo1:qi[1] + h, k]
        a_ = ax[1, i + 1]; a_.imshow(patch.T, cmap='gray', origin='lower')
        a_.plot(q[0] - lo0, q[1] - lo1, 'o', mfc='none', mec='r', ms=14)
        a_.plot(pl[i][0] - lo0, pl[i][1] - lo1, 'c+', ms=8)
        a_.set_title(f"m{i} V={rec['void_strength'][i]:.2f} z={k}", fontsize=8)
        if k + 1 < T.shape[2]:
            b = ax[0, i + 1]; b.imshow(T[lo0:qi[0] + h, lo1:qi[1] + h, k + 1].T, cmap='gray', origin='lower')
            b.plot(q[0] - lo0, q[1] - lo1, 'o', mfc='none', mec='y', ms=14); b.set_title(f'm{i} slice {k + 1}', fontsize=8)
    fig.suptitle(f"{rec['subject']}  src={rec.get('source')} n={rec.get('n_src')}  rms={rec.get('rms_mm', float('nan')):.2f} mm  "
                 f"rot={rec.get('rot_deg', float('nan')):.1f} deg  margin={rec.get('margin', float('nan')):.2f}", fontsize=9)
    fig.tight_layout(); fig.savefig(out_png, dpi=80); plt.close(fig)


# acceptance rule, set on the tuning half of the study subjects (see validate)
RULE = dict(rot_max=13.0, rms_max=1.0, min_clear=2)


def accept(rec, rule=RULE):
    return (rec is not None and rec['rot_deg'] <= rule['rot_max'] and rec['rms_mm'] <= rule['rms_max']
            and rec['n_clear_void'] >= rule['min_clear'])


# ---------------------------------------------------------------- validation
def validate(a):
    import pandas as pd
    subs = open(P + '/subjects.txt').read().split()
    rows = []
    os.makedirs(a.out, exist_ok=True)
    for sid in subs:
        dd = sorted(glob.glob(f'{P}/dicoms/*_{sid}'))
        if not dd or not os.path.isdir(dd[-1] + '/CT'):
            continue
        out = os.path.join(a.out, f'{sid}.json')
        if os.path.exists(out):
            rec = json.load(open(out))
        else:
            ct = load_ct(dd[-1] + '/CT')
            src, cinfo = ct_markers(ct[0], ct[1]) if ct else ([], {})
            rec = dict(subject=sid, source='ct', n_src=len(src), ct=cinfo)
            if len(src) >= 2:
                r = label_subject(src, f'{P}/images/{sid}.nii', f'{P}/cache/voidmap/{sid}.npz')
                if r:
                    rec.update(r)
            json.dump(rec, open(out, 'w'), indent=1, default=float)
        # compare with the corrected reference (QC-excluded markers dropped)
        ref = json.load(open(f'{P}/labelfix/refined/{sid}.json')); ex = set(ref.get('excluded', []))
        z = np.array(nib.load(f'{P}/images/{sid}.nii').header.get_zooms()[:3])
        R = np.array([v for i, v in enumerate(ref['markers_final_vox']) if i not in ex]) * z
        row = dict(subject=sid, half=0 if subs.index(sid) < len(subs) // 2 else 1, n_ref=len(R), n_src=rec['n_src'],
                   **{k: rec.get(k) for k in ['rms_mm', 'margin', 'n_clear_void', 'rot_deg', 'score', 'n_mr_blobs', 'triple_hyps']})
        if 'markers_final_vox' in rec:
            Q = np.array(rec['markers_final_vox']) * z
            D = np.linalg.norm(Q[:, None] - R[None], axis=2)
            row['dist_ref'] = D.min(0).tolist()   # per reference marker: nearest auto marker
            row['extra'] = int((D.min(1) > 5).sum())
        rows.append(row)
        print(sid, row.get('n_src'), row.get('rms_mm'), row.get('margin'), np.round(row.get('dist_ref', []), 2), flush=True)
    df = pd.DataFrame(rows); df.to_csv(os.path.join(a.out, 'validation.csv'), index=False)


def prepare_t1(series_dir, out, uid=None):
    import SimpleITK as sitk
    sys.path.insert(0, os.path.join(P, 'raw', 'scripts', 'preprocessing'))
    from homogeneity_correction import apply_homogeneity_correction
    r = sitk.ImageSeriesReader()
    uid = uid or max(r.GetGDCMSeriesIDs(series_dir), key=lambda u: len(r.GetGDCMSeriesFileNames(series_dir, u)))
    r.SetFileNames(r.GetGDCMSeriesFileNames(series_dir, uid)); im = r.Execute()
    arr = sitk.GetArrayFromImage(im).transpose(2, 1, 0).astype(np.float32)
    A = np.eye(4); A[:3, :3] = np.array(im.GetDirection()).reshape(3, 3) * np.array(im.GetSpacing())
    A[:3, 3] = im.GetOrigin(); A = np.diag([-1, -1, 1, 1]) @ A          # LPS -> RAS
    F = np.eye(4); F[1, 1] = -1; F[1, 3] = arr.shape[1] - 1                # dcm2niix row order
    arr = arr[:, ::-1]; A = A @ F
    raw = nib.Nifti1Image(arr, A)
    nib.save(nib.Nifti1Image(apply_homogeneity_correction(np.asarray(raw.dataobj, np.float64)), A, raw.header), out)
    return uid, arr.shape


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest='cmd', required=True)
    pr = sp.add_parser('prepare'); pr.add_argument('--series-dir', required=True); pr.add_argument('--uid')
    pr.add_argument('--out', required=True)
    v = sp.add_parser('validate'); v.add_argument('--out', required=True)
    p = sp.add_parser('apply')
    p.add_argument('--dicom', required=True, help='archive dir (searched recursively for CT / RTSTRUCT)')
    p.add_argument('--t1', required=True, help='N4-corrected T1 NIfTI'); p.add_argument('--sid', required=True)
    p.add_argument('--source', choices=['auto', 'ct', 'rs'], default='auto')
    p.add_argument('--out', required=True)
    a = ap.parse_args()
    if a.cmd == 'validate':
        return validate(a)
    if a.cmd == 'prepare':
        print(*prepare_t1(a.series_dir, a.out, a.uid)); return
    rec = dict(subject=a.sid)
    src = []; ct = None
    if a.source in ('auto', 'ct'):
        ct = load_ct(a.dicom)
        if ct:
            src, info = ct_markers(ct[0], ct[1]); rec.update(source='ct', ct=info, ct_meta=ct[2])
    if len(src) < 2 and a.source in ('auto', 'rs'):
        src, info = rs_markers(a.dicom); rec.update(source='rtstruct', rs=info)
    rec['n_src'] = len(src); rec['src_lps'] = [list(map(float, s['lps'])) for s in src]
    if len(src) >= 2:
        r = label_subject(src, a.t1)
        if r:
            rec.update(r)
    rec['accepted'] = bool(accept(rec, RULE)) if 'rms_mm' in rec else False
    os.makedirs(a.out, exist_ok=True)
    json.dump(rec, open(os.path.join(a.out, f'{a.sid}.json'), 'w'), indent=1, default=float)
    if 'markers_final_vox' in rec:
        qc_png(rec, a.t1, os.path.join(a.out, f'{a.sid}_qc.png'), ct[:2] if rec.get('source') == 'ct' else None)
    print(a.sid, rec.get('accepted'), rec.get('source'), rec['n_src'], rec.get('rms_mm'), rec.get('margin'), rec.get('n_clear_void'))


if __name__ == '__main__':
    main()
