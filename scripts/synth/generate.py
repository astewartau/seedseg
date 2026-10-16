"""Generate synthetic training cases: real prostate T1w + simulated gold markers
(labelled class 1) + calcification/blood-product-like distractors (unlabelled).

Usage
  python scripts/synth/generate.py --subjects all --variant corrected --k 2 --seed 0 \
      [--label-mode sphere|footprint] [--nproc 16] [--set readout_axis=0 --set bw_px=250]

Outputs (root = /scratch/user/uqaste15/prostate-v2/synthetic/{variant})
  images/{sid}_syn{k}.nii.gz   int16 (same header/scaling convention as the source)
  labels/{sid}_syn{k}.nii.gz   uint8: source label variant + synthetic markers (class 1)
  params/{sid}_syn{k}.json     every simulation parameter (reproducible from the JSON)

Each case is deterministic given (--seed, sid, k).
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import subprocess
import sys
import time
from multiprocessing import Pool

import nibabel as nib
import numpy as np
from scipy import ndimage as ndi

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import real_voids as R  # noqa: E402
from insert import insert_sources, PATCH_HALF  # noqa: E402
from physics import Acquisition, Grid, axis_from_angles, b0_dir_voxel_frame  # noqa: E402

ROOT = "/scratch/user/uqaste15/prostate-v2"

# Acquisition settings validated against real voids (validate.py --variant final):
# off-resonance slice displacement with RF bandwidth 1300 Hz, positive polarity.
DEFAULT_ACQ = dict(rf_bw=1300.0, sign_ss=1.0)

# Polar angle (to B0) prior for synthetic markers: the pooled posterior of the
# validation fit (validate.py --variant final) on a 10-degree grid; each draw is the
# grid angle + U(-5, 5) deg, azimuth uniform.
THETA_GRID = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
THETA_PROB = [0.103, 0.101, 0.125, 0.114, 0.105, 0.108, 0.095, 0.090, 0.074, 0.084]

DEFAULTS = dict(
    n_markers=(1, 3),          # synthetic markers per case (inclusive range)
    n_calcs=(0, 4),            # distractors per case
    marker_dchi=(-35.0, -15.0),  # ppm, uniform; nominal gold (-34) - tissue (-9) = -25
    theta_grid=THETA_GRID, theta_prob=THETA_PROB,
    marker_radius=0.5, marker_length=3.0,  # mm
    min_dist_real=10.0,        # mm from any real-marker label voxel
    min_dist_syn=8.0,          # mm between synthetic markers (centres)
    erode_mm=1.5,              # marker centres at least this far inside the prostate
    p_context=dict(random=0.5, boundary=0.25, bright=0.25),
    boundary_band=(1.5, 4.0),  # mm from the gland edge for 'boundary' context
    bright_factor=1.6, bright_dist=4.0,  # 'bright' = within 4 mm of voxel > 1.6 x CTV median
    calc_diam=(0.5, 3.0),      # mm equivalent diameter
    calc_dchi_dia=(-10.0, -1.0), calc_dchi_para=(1.0, 10.0),  # ppm (log-uniform magnitude)
    p_paramagnetic=0.25,
    calc_rho_dia=(0.0, 0.3), calc_rho_para=(0.3, 1.0),
    calc_min_dist_marker=6.0,  # mm from synthetic marker centres and real marker voxels
    sphere_radius=3.0,         # mm, label sphere for synthetic markers
    footprint_thr=0.5,         # |A| threshold for 'footprint' label mode
    n_coils=10, noise_corr=(0.38, 0.22),
)


def git_rev():
    try:
        return subprocess.check_output(["git", "-C", os.path.dirname(os.path.abspath(__file__)),
                                        "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def case_rng(seed, sid, k):
    ss = np.random.SeedSequence([int(seed), int(sid.lstrip("z")), int(k)])
    return np.random.default_rng(ss), ss.entropy


def random_axis(rng):
    v = rng.standard_normal(3)
    return v / np.linalg.norm(v)


def loguniform(rng, lo, hi):
    s = np.sign(lo)
    a, b = sorted([abs(lo), abs(hi)])
    return float(s * np.exp(rng.uniform(np.log(a), np.log(b))))


def random_rotation(rng):
    q = rng.standard_normal(4)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def calc_spec(rng, center, P):
    d = rng.uniform(*P["calc_diam"])
    n_ell = int(rng.integers(1, 4))
    ells = []
    for e in range(n_ell):
        semi = (d / 2) * rng.uniform(0.6, 1.4, 3) * (1.0 if e == 0 else rng.uniform(0.4, 0.8))
        off = np.zeros(3) if e == 0 else rng.uniform(-0.5, 0.5, 3) * d
        ells.append(dict(offset_mm=off.tolist(), semi_axes_mm=semi.tolist(),
                         rot=random_rotation(rng).tolist()))
    para = rng.uniform() < P["p_paramagnetic"]
    if para:
        dchi = loguniform(rng, *P["calc_dchi_para"])
        rho = rng.uniform(*P["calc_rho_para"])
    else:
        dchi = loguniform(rng, *P["calc_dchi_dia"])
        rho = rng.uniform(*P["calc_rho_dia"])
    return dict(type="calc", center_vox=list(map(float, center)), dchi_ppm=dchi, rho=float(rho),
                ellipsoids=ells, equiv_diam_mm=float(d), paramagnetic=bool(para))


def candidate_maps(img, lab, vox, P):
    prostate = (lab == 2) | (lab == 1)
    din = ndi.distance_transform_edt(prostate, sampling=vox)
    dreal = ndi.distance_transform_edt(lab != 1, sampling=vox) if np.any(lab == 1) else np.full(lab.shape, 1e9)
    inb = np.zeros(lab.shape, bool)
    h = np.array(PATCH_HALF) + 1
    inb[h[0]:-h[0], h[1]:-h[1], h[2]:-h[2]] = True
    ref = R.ctv_median(img, lab)
    bright = (img > P["bright_factor"] * ref) & ndi.binary_dilation(prostate, iterations=6)
    dbright = ndi.distance_transform_edt(~bright, sampling=vox) if bright.any() else np.full(lab.shape, 1e9)
    base = inb & (din >= P["erode_mm"]) & (dreal >= P["min_dist_real"])
    return dict(
        random=base,
        boundary=base & (din <= P["boundary_band"][1]) & (din >= P["boundary_band"][0]),
        bright=base & (dbright <= P["bright_dist"]),
        calc=inb & prostate & (din >= 0.5) & (dreal >= P["calc_min_dist_marker"] - P["sphere_radius"]),
    ), ref


def pick(rng, mask, taken, min_d, vox):
    idx = np.argwhere(mask)
    if len(idx) == 0:
        return None
    for _ in range(200):
        p = idx[rng.integers(len(idx))] + rng.uniform(-0.5, 0.5, 3)
        if all(np.linalg.norm((p - t) * vox) >= min_d for t in taken):
            return p
    return None


def sphere_mask(shape, c, r_mm, vox):
    hw = [int(np.ceil(r_mm / v)) + 1 for v in vox]
    lo = [max(int(round(x)) - h, 0) for x, h in zip(c, hw)]
    hi = [min(int(round(x)) + h + 1, n) for x, h, n in zip(c, hw, shape)]
    g = np.meshgrid(*[np.arange(a, b) for a, b in zip(lo, hi)], indexing="ij")
    rr = np.sqrt(sum(((gg - cc) * v) ** 2 for gg, cc, v in zip(g, c, vox)))
    m = np.zeros(shape, bool)
    m[tuple(slice(a, b) for a, b in zip(lo, hi))] = rr <= r_mm
    return m


def make_case(sid, k, variant, seed, out_root, label_mode, acq_overrides, P):
    t0 = time.time()
    rng, entropy = case_rng(seed, sid, k)
    im = nib.load(f"{ROOT}/images/{sid}.nii")
    img = np.asarray(im.dataobj, dtype=np.float32)
    lab_im = nib.load(f"{ROOT}/labels/{variant}/{sid}.nii.gz")
    lab = np.asarray(lab_im.dataobj).astype(np.uint8)
    vox = np.array(im.header.get_zooms()[:3], float)
    acq = Acquisition(voxel=tuple(vox), b0_dir=tuple(b0_dir_voxel_frame(im.affine)), **acq_overrides)
    grid = Grid()
    maps, ref = candidate_maps(img, lab, vox, P)
    sigma = R.local_noise_sigma(img, ndi.binary_dilation(lab == 2, iterations=3))

    # ---- synthetic markers ----
    n_m = int(rng.integers(P["n_markers"][0], P["n_markers"][1] + 1))
    ctx_names = list(P["p_context"])
    ctx_p = np.array([P["p_context"][c] for c in ctx_names])
    markers, taken = [], []
    for _ in range(n_m):
        ctx = ctx_names[rng.choice(len(ctx_names), p=ctx_p / ctx_p.sum())]
        c = pick(rng, maps[ctx], taken, P["min_dist_syn"], vox)
        if c is None and ctx != "random":
            ctx, c = "random(fallback)", pick(rng, maps["random"], taken, P["min_dist_syn"], vox)
        if c is None:
            continue
        taken.append(c)
        b = np.asarray(acq.b0_dir) / np.linalg.norm(acq.b0_dir)
        if P["theta_prob"] is None:
            u = random_axis(rng)
        else:
            tp = np.asarray(P["theta_prob"], float)
            th0 = P["theta_grid"][rng.choice(len(tp), p=tp / tp.sum())]
            th = float(np.clip(th0 + rng.uniform(-5, 5), 0, 90))
            u = axis_from_angles(th, rng.uniform(0, 360), b)
        dchi = rng.uniform(*P["marker_dchi"])
        markers.append(dict(type="marker", center_vox=c.tolist(), axis=u.tolist(), dchi_ppm=float(dchi),
                            radius_mm=P["marker_radius"], length_mm=P["marker_length"], context=ctx,
                            theta_deg=float(np.degrees(np.arccos(abs(u @ b))))))

    # ---- distractors ----
    n_c = int(rng.integers(P["n_calcs"][0], P["n_calcs"][1] + 1))
    calcs = []
    cmask = maps["calc"].copy()
    for m in markers:
        cmask &= ~sphere_mask(cmask.shape, m["center_vox"], P["calc_min_dist_marker"], vox)
    ctaken = []
    for _ in range(n_c):
        c = pick(rng, cmask, ctaken, 2.0 + P["calc_diam"][1], vox)
        if c is None:
            continue
        ctaken.append(c)
        calcs.append(calc_spec(rng, c, P))

    specs = markers + calcs
    out, A, infos = insert_sources(img, specs, acq, grid, sigma, rng,
                                   noise_corr=tuple(P["noise_corr"]), n_coils=P["n_coils"])

    # ---- labels ----
    new_lab = lab.copy()
    for m, info in zip(markers, infos[:len(markers)]):
        if info is None:
            continue
        if label_mode == "sphere":
            new_lab[sphere_mask(lab.shape, m["center_vox"], P["sphere_radius"], vox)] = 1
        else:  # footprint: thresholded void connected to the marker + gold occupancy
            sl = info["slices"]
            fp = (np.abs(info["A"]) < P["footprint_thr"]) | (info["occ"] > 0)
            cc, _ = ndi.label(fp)
            ci = tuple(int(round(x)) - s.start for x, s in zip(m["center_vox"], sl))
            keep = cc == cc[ci] if cc[ci] > 0 else fp
            sub = new_lab[sl]
            sub[keep] = 1
        m["void_area_mm2_minslice"] = float(
            (np.abs(info["A"]) < P["footprint_thr"]).sum(axis=(0, 1)).max() * vox[0] * vox[1])

    os.makedirs(f"{out_root}/images", exist_ok=True)
    os.makedirs(f"{out_root}/labels", exist_ok=True)
    os.makedirs(f"{out_root}/params", exist_ok=True)
    name = f"{sid}_syn{k}"
    oim = nib.Nifti1Image(out.astype(np.float32), im.affine, header=im.header)
    oim.set_data_dtype(im.header.get_data_dtype())  # int16 with auto scaling, like the source
    nib.save(oim, f"{out_root}/images/{name}.nii.gz")
    olab = nib.Nifti1Image(new_lab, lab_im.affine, header=lab_im.header)
    olab.set_data_dtype(np.uint8)
    nib.save(olab, f"{out_root}/labels/{name}.nii.gz")
    rec = dict(case=name, subject=sid, k=int(k), variant=variant, seed=int(seed), seed_entropy=str(entropy),
               label_mode=label_mode, git=git_rev(), acquisition=acq.to_dict(),
               grid=dataclasses.asdict(grid), noise=dict(sigma=float(sigma), ctv_median=float(ref),
                                                         n_coils=P["n_coils"], corr=list(P["noise_corr"])),
               params={k_: (list(v) if isinstance(v, tuple) else v) for k_, v in P.items()},
               markers=markers, calcs=calcs, runtime_s=time.time() - t0)
    with open(f"{out_root}/params/{name}.json", "w") as f:
        json.dump(rec, f, indent=1)
    return name, len(markers), len(calcs), rec["runtime_s"]


def _job(a):
    try:
        return make_case(*a)
    except Exception as e:  # keep the batch going; report at the end
        return a[0], -1, -1, repr(e)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subjects", default="all", help="'all', a comma list, or a file with one id per line")
    ap.add_argument("--variant", default="corrected")
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--label-mode", choices=["sphere", "footprint"], default="sphere")
    ap.add_argument("--out", default=None, help="default: prostate-v2/synthetic/{variant}")
    ap.add_argument("--nproc", type=int, default=16)
    ap.add_argument("--set", action="append", default=[],
                    help="Acquisition override key=value (JSON value), e.g. readout_axis=0")
    ap.add_argument("--param", action="append", default=[],
                    help="generator parameter override key=value (JSON value), e.g. n_markers=[2,3]")
    a = ap.parse_args()
    if a.subjects == "all":
        sids = sorted(f[:-4] for f in os.listdir(f"{ROOT}/images") if f.endswith(".nii"))
        sids = [s for s in sids if os.path.exists(f"{ROOT}/labels/{a.variant}/{s}.nii.gz")]
    elif os.path.isfile(a.subjects):
        sids = [l.strip() for l in open(a.subjects) if l.strip()]
    else:
        sids = a.subjects.split(",")
    acq_over = dict(DEFAULT_ACQ)
    for kv in a.set:
        k, v = kv.split("=", 1)
        v = json.loads(v)
        acq_over[k] = tuple(v) if isinstance(v, list) else v
    P = dict(DEFAULTS)
    for kv in a.param:
        k, v = kv.split("=", 1)
        if k not in P:
            raise SystemExit(f"unknown parameter {k}")
        P[k] = json.loads(v)
    out_root = a.out or f"{ROOT}/synthetic/{a.variant}"
    jobs = [(s, k, a.variant, a.seed, out_root, a.label_mode, acq_over, P) for s in sids for k in range(a.k)]
    t0 = time.time()
    with Pool(a.nproc) as pool:
        res = pool.map(_job, jobs, chunksize=1)
    bad = [r for r in res if r[1] < 0]
    rt = [r[3] for r in res if r[1] >= 0]
    print(f"{len(res) - len(bad)} cases written to {out_root} in {time.time() - t0:.0f}s "
          f"(per case median {np.median(rt):.1f}s, single-core)")
    for r in bad:
        print("FAILED", r)


if __name__ == "__main__":
    main()
