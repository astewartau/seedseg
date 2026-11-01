#!/usr/bin/env python3
"""Step 1 of the synthetic demo data: per-subject tissue label maps on a common grid.

For each subject the clinical RTSTRUCT organ contours (body, bone and femoral heads,
bladder, rectum, CTV, seminal vesicles, penile bulb, spacer, gas override) are
evaluated on a 1 mm isotropic grid in the MR voxel-aligned frame, centred on the CTV
centroid. Each grid point is mapped MR -> CT with the clinical registration and the
accepted rigid refinement of the v2 labels (labelfix/markers, labelfix/refined), and
tested against the planar contours on the nearest CT contour plane (even-odd rule).

The N4-corrected T1 is sampled at the same points. Classes without a contour are split
by intensity relative to the CTV median r = I / median(I in CTV):
  inside body, outside organs : gas (r < GAS_THR), else three soft-tissue classes
                                (dark / mid / bright) by per-subject 1D k-means
  bone                        : cortical bone / marrow by per-subject 2-means
  rectum                      : gas (r < GAS_THR) / rectum

Outputs (private, stay on Bunya): {out}/{sid}_lab.nii.gz (uint8 classes, 255 outside
the MR field of view), {out}/{sid}_img.nii.gz (float32, r units) and {out}/{sid}.json.
"""
import argparse, glob, json, os, re, subprocess, tempfile
import numpy as np
import nibabel as nib
import scipy.ndimage as ndi
from scipy.spatial.transform import Rotation

ROOT = "/scratch/user/uqaste15/prostate-v2"
DICOM = "/QRISdata/Q0748/data/prostate/dicoms-organized"

CLASSES = ["air", "soft_dark", "soft_mid", "soft_bright", "marrow", "cortex", "bladder", "rectum", "gas",
           "prostate", "seminal", "bulb", "spacer"]
C = {n: i for i, n in enumerate(CLASSES)}
OUTSIDE_FOV = 255
GRID_MM = (240, 240, 84)          # extent of the common grid (x, y, z), 1 mm spacing
GAS_THR = 0.45

# ROI name patterns (case-insensitive, full match after stripping); first hit wins
ROI_PATTERNS = {
    "body": [r"body\d*", r"external", r"outline", r"skin"],
    "bone": [r"bone", r"bones", r"pelvic ?bones?", r"pelvis"],
    "femur_l": [r"femur_?head_?l", r"femoral ?head ?l(eft)?", r"femur ?l"],
    "femur_r": [r"femur_?head_?r", r"femoral ?head ?r(ight)?", r"femur ?r"],
    "bladder": [r"bladder"],
    "rectum": [r"rectum"],
    "seminal": [r"seminalves", r"seminal ?vesicles?", r"sv"],
    "bulb": [r"penile ?bulb"],
    "spacer": [r"space ?oar", r"spacer", r"hydrogel"],
    "gas": [r"z?gas_?override", r"gas"],
}


def find_roi(names, key):
    for pat in ROI_PATTERNS[key]:
        for num, n in names.items():
            if re.fullmatch(pat, n.strip(), re.I):
                return num, n
    return None, None


def roi_planes(rs, num):
    """{z: [polygon (k, 2)]} in the CT frame for a CLOSED_PLANAR ROI."""
    cs = next((c for c in rs.ROIContourSequence if c.ReferencedROINumber == num), None)
    planes = {}
    for cc in getattr(cs, "ContourSequence", []) if cs is not None else []:
        p = np.array(cc.ContourData, float).reshape(-1, 3)
        if len(p) < 3:
            continue
        planes.setdefault(round(float(p[:, 2].mean()), 2), []).append(p[:, :2])
    return planes


def inside_roi(planes, pts):
    from matplotlib.path import Path
    """Even-odd point-in-ROI test for CT-frame points (n, 3) on the nearest contour plane;
    points beyond half a plane spacing outside the ROI's z range are outside."""
    out = np.zeros(len(pts), bool)
    if not planes:
        return out
    zs = np.array(sorted(planes))
    dz = np.median(np.diff(zs)) if len(zs) > 1 else 2.0
    j = np.clip(np.searchsorted(zs, pts[:, 2]), 0, len(zs) - 1)
    jm = np.clip(j - 1, 0, len(zs) - 1)
    j = np.where(np.abs(zs[jm] - pts[:, 2]) < np.abs(zs[j] - pts[:, 2]), jm, j)
    ok = np.abs(zs[j] - pts[:, 2]) <= dz / 2 + 1e-3
    for k in np.unique(j[ok]):
        sel = np.nonzero(ok & (j == k))[0]
        xy = pts[sel, :2]
        acc = np.zeros(len(sel), bool)
        lo, hi = xy.min(0), xy.max(0)
        for poly in planes[zs[k]]:
            pl, ph = poly.min(0), poly.max(0)
            if np.any(ph < lo) or np.any(pl > hi):
                continue
            acc ^= Path(poly).contains_points(xy)
        out[sel] = acc
    return out


def kmeans1d(x, k, iters=30):
    """Labels 0..k-1 ordered by increasing centre (1D Lloyd from quantile init)."""
    c = np.quantile(x, (np.arange(k) + 0.5) / k)
    for _ in range(iters):
        b = (c[1:] + c[:-1]) / 2
        lab = np.searchsorted(b, x)
        c = np.array([x[lab == j].mean() if np.any(lab == j) else c[j] for j in range(k)])
    return np.searchsorted((c[1:] + c[:-1]) / 2, x)


def extract_rs(archive, rs_name, dst):
    tar = f"{DICOM}/{archive}.tar.gz"
    subprocess.run(["tar", "xzf", tar, "-C", dst, "--wildcards", f"*/RS/{rs_name}"], check=True)
    return glob.glob(f"{dst}/*/RS/{rs_name}")[0]


def grid_points(ctr_mm):
    """1 mm grid (x, y, z order) around ctr_mm in the MR voxel-aligned mm frame."""
    ax = [np.arange(n) - (n - 1) / 2 + c for n, c in zip(GRID_MM, ctr_mm)]
    G = np.stack(np.meshgrid(*ax, indexing="ij"), -1)
    return G.reshape(-1, 3)


def subject(sid, out):
    import pydicom
    mrec = json.load(open(f"{ROOT}/labelfix/markers/{sid}.json"))
    rrec = json.load(open(f"{ROOT}/labelfix/refined/{sid}.json"))
    im = nib.load(f"{ROOT}/images/{sid}.nii")
    vol = np.asarray(im.dataobj, np.float32)
    aff = im.affine
    z = np.array(im.header.get_zooms()[:3], float)
    lab_v2 = np.asarray(nib.load(f"{ROOT}/labels/corrected/{sid}.nii.gz").dataobj)
    ref = float(np.median(vol[lab_v2 == 2]))
    ctv_vox = np.array(ndi.center_of_mass(lab_v2 >= 1))
    ctr_mm = ctv_vox * z

    # rigid refinement (voxel-aligned mm frame): q = (p - c) Rm^T + c + t  =>  p = (q - c - t) Rm + c
    Rm = Rotation.from_euler("xyz", rrec["rot_deg"], degrees=True).as_matrix()
    t = np.array(rrec["trans_mm"]); c = np.array(rrec["centre_vox"]) * z
    if not rrec.get("accepted"):
        Rm, t = np.eye(3), np.zeros(3)
    reg = np.array(mrec["reg_matrix"])          # x_ct = reg @ x_mr (LPS)

    q = grid_points(ctr_mm)
    p_mm = (q - c - t) @ Rm + c
    v = p_mm / z                                # MR voxel coords
    ras = (aff @ np.c_[v, np.ones(len(v))].T).T[:, :3]
    lps = ras * np.array([-1, -1, 1])
    ct = (reg @ np.c_[lps, np.ones(len(lps))].T).T[:, :3]
    in_fov = np.all((v >= -0.5) & (v <= np.array(vol.shape) - 0.5), 1)
    # the scanner images carry a zero-valued frame at the in-plane edges: not anatomy
    zero = ndi.binary_dilation(vol <= 1e-3 * np.abs(vol).max(), iterations=1)
    in_fov &= ndi.map_coordinates(zero.astype(np.float32), v.T, order=1, mode="nearest") < 0.01

    with tempfile.TemporaryDirectory(dir="/scratch/user/uqaste15/demo-synth/tmp") as td:
        rs = pydicom.dcmread(extract_rs(mrec["archive"], mrec["rtstruct"], td))
    names = {r.ROINumber: r.ROIName for r in rs.StructureSetROISequence}
    found, masks = {}, {}
    for key in ROI_PATTERNS:
        num, name = find_roi(names, key)
        if num is not None:
            masks[key] = inside_roi(roi_planes(rs, num), ct); found[key] = name
    ctv_num = next(k for k, n in names.items() if n == mrec["ctv_roi"])
    masks["prostate"] = inside_roi(roi_planes(rs, ctv_num), ct); found["prostate"] = mrec["ctv_roi"]

    shape = GRID_MM
    r = ndi.map_coordinates(vol, v.T, order=1, mode="nearest").reshape(shape) / ref
    M = {k: m.reshape(shape) for k, m in masks.items()}
    body = M.get("body", np.ones(shape, bool))
    lab = np.full(shape, C["air"], np.uint8)
    soft = body & (r >= GAS_THR)
    lab[body & ~soft] = C["gas"]
    k3 = kmeans1d(r[soft], 3)
    lab[soft] = np.array([C["soft_dark"], C["soft_mid"], C["soft_bright"]], np.uint8)[k3]
    bone = M.get("bone", np.zeros(shape, bool)) | M.get("femur_l", False) | M.get("femur_r", False)
    lab[bone] = np.array([C["cortex"], C["marrow"]], np.uint8)[kmeans1d(r[bone], 2)]
    for key in ["bladder", "rectum", "bulb", "spacer", "seminal", "prostate"]:
        if key in M:
            lab[M[key]] = C[key]
    if "rectum" in M:
        g = M["rectum"] & (r < GAS_THR)
        lab[g] = C["gas"]
    if "gas" in M:
        lab[M["gas"] & (lab != C["prostate"])] = C["gas"]
    lab[~in_fov.reshape(shape)] = OUTSIDE_FOV

    A = np.eye(4); A[:3, 3] = -(np.array(shape) - 1) / 2   # grid centre at the origin (mm)
    nib.save(nib.Nifti1Image(lab, A), f"{out}/{sid}_lab.nii.gz")
    nib.save(nib.Nifti1Image(r.astype(np.float32), A), f"{out}/{sid}_img.nii.gz")
    counts = {n: int((lab == i).sum()) for i, n in enumerate(CLASSES)}
    gas_out = int(((lab == C["gas"]) & ~M.get("rectum", np.zeros(shape, bool))).sum())
    rec = dict(subject=sid, rois=found, gas_outside_rectum=gas_out, ctv_median=ref, ctv_centre_vox=ctv_vox.tolist(),
               mr_shape=list(vol.shape), mr_affine=aff.tolist(), refinement_accepted=bool(rrec.get("accepted")),
               voxels=counts, fov_fraction=float(in_fov.mean()))
    json.dump(rec, open(f"{out}/{sid}.json", "w"), indent=1)
    return sid, found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", required=True, help="file with one id per line")
    ap.add_argument("--out", required=True)
    ap.add_argument("--nproc", type=int, default=8)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    os.makedirs("/scratch/user/uqaste15/demo-synth/tmp", exist_ok=True)
    sids = [l.strip() for l in open(a.subjects) if l.strip()]
    from multiprocessing import Pool
    with Pool(a.nproc) as p:
        for res in p.imap_unordered(_job, [(s, a.out) for s in sids]):
            print(*res, flush=True)


def _job(x):
    try:
        return subject(*x)
    except Exception as e:
        return x[0], f"FAILED {e!r}"


if __name__ == "__main__":
    main()
