"""Validate simulated gold-marker voids against real voids.

Steps
  1. Real reference set: refined MR marker positions of all subjects
     (labelfix/refined/{sid}.json, void_strength >= --min-strength); the void centre is
     re-located near each position with the same code used for simulated voids.
  2. Simulation library: gold cylinders inserted into marker-free prostate locations
     of the same subjects, over a grid of polar angles theta (to B0), random azimuth,
     random sub-voxel position and a few delta-chi values; voids are located and
     described with exactly the same code as the real ones.
  3. Matching: for each real void, the nearest library entries (rotation-invariant
     per-slice radial profiles + in-plane elongation) give a posterior over
     (theta, delta-chi). The pooled posterior defines the fitted orientation
     distribution.
  4. A fresh simulated set drawn from the fitted distribution is compared to the real
     set (profiles, per-slice minimum, void area, elongation, KS tests, montages).

Usage
  python scripts/synth/validate.py --variant final --set rf_bw=1300 --set sign_ss=1
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import features as F  # noqa: E402
import real_voids as R  # noqa: E402
from insert import insert_sources  # noqa: E402
from physics import Acquisition, Grid, axis_from_angles, b0_dir_voxel_frame  # noqa: E402



def parse_sets(sets):
    """--set key=value pairs -> (Acquisition kwargs, noise_scale, noise_corr_axis0)."""
    acq_kw, nscale, ncorr = {}, 1.0, (0.38, 0.22)
    for kv in sets or []:
        k, v = kv.split("=", 1)
        try:
            v = json.loads(v)
        except json.JSONDecodeError:
            pass
        if k == "n_coils":
            acq_kw["_n_coils"] = int(v)
        elif k == "noise_scale":
            nscale = float(v)
        elif k == "noise_corr":
            ncorr = tuple(v) if isinstance(v, list) else (float(v), 0.0)
        else:
            acq_kw[k] = tuple(v) if isinstance(v, list) else v
    return acq_kw, nscale, ncorr

N_COILS = 10
THETAS = np.arange(0, 91, 10)
DCHIS = np.array([-15.0, -25.0, -35.0])

_CACHE = {}


REFINED = "/scratch/user/uqaste15/prostate-v2/labelfix/refined"
LABEL_VARIANT = "corrected"


def reference_subjects():
    """All subjects with refined (MR-void-based) marker centres."""
    return sorted(f[:-5] for f in os.listdir(REFINED) if f.endswith(".json"))


def refined_markers(sid):
    with open(f"{REFINED}/{sid}.json") as f:
        j = json.load(f)
    return [(np.array(c, float), float(s)) for c, s in
            zip(j.get("markers_final_vox") or [], j.get("void_strength") or [])]


def subject(sid):
    if sid not in _CACHE:
        im, img, lab = R.load_case(sid, LABEL_VARIANT)
        ref = R.ctv_median(img, lab)
        sig = R.local_noise_sigma(img, ndi.binary_dilation(lab == 2, iterations=3))
        _CACHE[sid] = dict(affine=im.affine, img=img, lab=lab, ref=ref, sigma=sig)
    return _CACHE[sid]


def background_sites(sid, n, rng, min_marker_mm=12.0, min_edge_mm=2.0):
    s = subject(sid)
    lab = s["lab"]
    vox = np.array(F.VOX)
    dmark = ndi.distance_transform_edt(lab != 1, sampling=vox)
    dedge = ndi.distance_transform_edt(lab == 2, sampling=vox)
    ok = (lab == 2) & (dmark >= min_marker_mm) & (dedge >= min_edge_mm)
    ok[:13], ok[-13:], ok[:, :13], ok[:, -13:] = False, False, False, False
    ok[:, :, :5], ok[:, :, -5:] = False, False
    idx = np.argwhere(ok)
    if len(idx) == 0:
        return []
    pick = idx[rng.choice(len(idx), size=n, replace=len(idx) < n)]
    return [p + rng.uniform(-0.5, 0.5, 3) for p in pick]


def sim_one(task):
    sid, center, theta, az, dchi, seed, cfg = task
    acq_kw, nscale, ncorr = cfg
    s = subject(sid)
    acq_kw = dict(acq_kw)
    n_coils = acq_kw.pop("_n_coils", N_COILS)
    acq = Acquisition(b0_dir=tuple(b0_dir_voxel_frame(s["affine"])), **acq_kw)
    rng = np.random.default_rng(seed)
    spec = dict(type="marker", center_vox=list(map(float, center)),
                axis=list(axis_from_angles(theta, az, acq.b0_dir)), dchi_ppm=float(dchi))
    out, A, info = insert_sources(s["img"], [spec], acq, Grid(), nscale * s["sigma"], rng,
                                  noise_corr=ncorr, n_coils=n_coils)
    # locate the void as for real markers: pseudo-label = 3.5 mm sphere at true centre
    v = locate(out, np.asarray(center), s["ref"])
    patch = R.extract_patch(out / s["ref"], v["center"])
    return dict(sid=sid, theta=float(theta), az=float(az), dchi=float(dchi), seed=int(seed),
                center=list(map(float, center)), found=v["center"], patch=patch.astype(np.float32))


def sphere_label(shape, c, radius_mm=3.5):
    lab = np.zeros(shape, np.uint8)
    hw = [int(np.ceil(radius_mm / v)) for v in F.VOX]
    lo = [max(int(round(x)) - h, 0) for x, h in zip(c, hw)]
    hi = [min(int(round(x)) + h + 1, n) for x, h, n in zip(c, hw, shape)]
    g = np.meshgrid(*[np.arange(a, b) for a, b in zip(lo, hi)], indexing="ij")
    rr = np.sqrt(sum(((gg - cc) * v) ** 2 for gg, cc, v in zip(g, c, F.VOX)))
    lab[tuple(slice(a, b) for a, b in zip(lo, hi))][rr <= radius_mm] = 1
    return lab


def locate(img, c, ref):
    """Void centre near nominal centre c, found exactly as for every real/simulated void."""
    return R.find_voids(img, sphere_label(img.shape, c), ref)[0]


def real_set(sids, min_strength=0.0):
    rows = []
    for sid in sids:
        s = subject(sid)
        for c, vs in refined_markers(sid):
            if vs < min_strength or np.any(c < [13, 13, 4]) or np.any(c > np.array(s["img"].shape) - [14, 14, 5]):
                continue
            v = locate(s["img"], c, s["ref"])
            rows.append(dict(sid=sid, center=v["center"], strength=vs,
                             patch=R.extract_patch(s["img"] / s["ref"], v["center"]).astype(np.float32)))
    return rows


def run_tasks(tasks, nproc):
    with Pool(nproc) as p:
        return p.map(sim_one, tasks, chunksize=4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="base", help="name used for output files")
    ap.add_argument("--set", action="append", default=[],
                    help="model override key=value (Acquisition field, noise_scale, noise_corr)")
    ap.add_argument("--thetas", default="0:91:10", help="library polar angles start:stop:step (deg)")
    ap.add_argument("--dchis", default="-15,-25,-35", help="library delta-chi values (ppm)")
    ap.add_argument("--reps", type=int, default=12, help="library sims per (theta, dchi)")
    ap.add_argument("--min-strength", type=float, default=0.45,
                    help="minimum void_strength of refined real markers used as references")
    ap.add_argument("--nproc", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="/scratch/user/uqaste15/prostate-v2/synthetic/validation")
    a = ap.parse_args()
    cfg = parse_sets(a.set)
    global THETAS, DCHIS
    THETAS = np.arange(*[int(x) for x in a.thetas.split(":")])
    DCHIS = np.array([float(x) for x in a.dchis.split(",")])
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    sids = reference_subjects()
    t0 = time.time()
    real = real_set(sids, a.min_strength)
    print(f"{len(real)} real voids from {len(sids)} subjects", flush=True)

    # ---- library ----
    tasks = []
    for th in THETAS:
        for dc in DCHIS:
            for r in range(a.reps):
                sid = sids[rng.integers(len(sids))]
                c = background_sites(sid, 1, rng)[0]
                tasks.append((sid, c, th, rng.uniform(0, 360), dc, int(rng.integers(2**31)), cfg))
    lib = run_tasks(tasks, a.nproc)
    print(f"library {len(lib)} sims in {time.time() - t0:.0f}s", flush=True)

    rd = [F.describe(r["patch"]) for r in real]
    ld = [F.describe(l["patch"]) for l in lib]
    RV = np.array([F.vector(d) for d in rd])
    LV = np.array([F.vector(d) for d in ld])
    lth = np.array([l["theta"] for l in lib])
    ldc = np.array([l["dchi"] for l in lib])
    D2 = ((RV[:, None, :] - LV[None, :, :]) ** 2).sum(-1)
    # posterior over (theta, dchi) per real void: softmax over classes of mean of the k
    # nearest within-class distances (k=3), temperature = median nn distance
    K = 3
    classes = [(th, dc) for th in THETAS for dc in DCHIS]
    cls_d = np.zeros((len(real), len(classes)))
    for ci, (th, dc) in enumerate(classes):
        m = (lth == th) & (ldc == dc)
        cls_d[:, ci] = np.sort(D2[:, m], axis=1)[:, :K].mean(1)
    T = np.median(cls_d.min(1))
    post = np.exp(-(cls_d - cls_d.min(1, keepdims=True)) / T)
    post /= post.sum(1, keepdims=True)
    pooled = post.mean(0).reshape(len(THETAS), len(DCHIS))
    best = np.array(classes)[cls_d.argmin(1)]

    # ---- fresh matched set drawn from pooled posterior ----
    p = pooled.ravel() / pooled.sum()
    draws = rng.choice(len(classes), size=len(real), p=p)
    mtasks = []
    for i, ci in enumerate(draws):
        th0, dc = classes[ci]
        th = float(np.clip(th0 + rng.uniform(-5, 5), 0, 90))
        sid = sids[rng.integers(len(sids))]
        c = background_sites(sid, 1, rng)[0]
        mtasks.append((sid, c, th, rng.uniform(0, 360), dc, int(rng.integers(2**31)), cfg))
    matched = run_tasks(mtasks, a.nproc)
    md = [F.describe(m["patch"]) for m in matched]

    res = summarise(rd, md, ld, lth, ldc)
    res.update(variant=a.variant, config=dict(acq=cfg[0], noise_scale=cfg[1], noise_corr=cfg[2]), n_real=len(real), n_lib=len(lib), runtime_s=time.time() - t0,
               pooled_posterior=dict(thetas=THETAS.tolist(), dchis=DCHIS.tolist(), p=pooled.tolist()),
               best_class_counts=pd.Series([f"{int(t)}|{int(d)}" for t, d in best]).value_counts().to_dict(),
               nn_dist_real_to_lib_median=float(np.sqrt(np.median(D2.min(1)))),
               nn_dist_lib_to_lib_median=float(np.sqrt(np.median(np.sort(
                   ((LV[:, None] - LV[None]) ** 2).sum(-1), 1)[:, 1]))))
    with open(f"{a.out}/metrics_{a.variant}.json", "w") as f:
        json.dump(res, f, indent=1)
    np.savez_compressed(f"{a.out}/patches_{a.variant}.npz",
                        real=np.array([r["patch"] for r in real]),
                        matched=np.array([m["patch"] for m in matched]),
                        lib=np.array([l["patch"] for l in lib]), lib_theta=lth, lib_dchi=ldc,
                        matched_theta=np.array([m["theta"] for m in matched]),
                        matched_dchi=np.array([m["dchi"] for m in matched]),
                        post=post, real_sid=np.array([r["sid"] for r in real]),
                        real_center=np.array([r["center"] for r in real]))
    print(json.dumps({k: v for k, v in res.items() if k not in ("pooled_posterior",)}, indent=1))


def summarise(rd, md, ld, lth, ldc):
    def arr(ds, key):
        return np.array([d[key] for d in ds])

    out = {}
    for key in ["min", "elong", "nslices", "vol"]:
        r, m = arr(rd, key), arr(md, key)
        ks = stats.ks_2samp(r, m)
        out[key] = dict(real_median=float(np.median(r)), real_iqr=np.percentile(r, [25, 75]).tolist(),
                        sim_median=float(np.median(m)), sim_iqr=np.percentile(m, [25, 75]).tolist(),
                        ks_stat=float(ks.statistic), ks_p=float(ks.pvalue))
    ra, ma = arr(rd, "area"), arr(md, "area")
    out["area_per_slice_mm2"] = dict(real_median=np.median(ra, 0).tolist(), sim_median=np.median(ma, 0).tolist())
    rr, mr = arr(rd, "radial"), arr(md, "radial")
    out["radial_minslice"] = dict(real_median=np.median(rr[:, :, F.NSL // 2], 0).tolist(),
                                  sim_median=np.median(mr[:, :, F.NSL // 2], 0).tolist())
    out["profile_rmse"] = float(np.sqrt(np.mean((np.median(rr, 0) - np.median(mr, 0)) ** 2)))
    return out


if __name__ == "__main__":
    main()
