#!/usr/bin/env python3
"""Step 5: the two release gates for synthetic demo cases.

Works:  the released v2 ensemble (ONNX, seedseg_onnx.py) on every case, both the N4-like
        image (--no-n4 path) and the shaded raw image (default path with N4). A marker
        is found when a top-3 detection lies within HIT_MM of its centre; every other
        top-3 detection is a false positive; extra candidate components (rank > 3) are reported.
Safe:   the case's 1 mm class map is compared with every real subject's anatomy (both
        on CTV-centred grids, no registration, which favours finding similarity). For
        each synthetic case the highest mean Dice to any real subject must stay below
        the 95th percentile of the real nearest-neighbour Dice (each real subject vs the
        most similar other real subject).
"""
import argparse, glob, json, os, sys
import numpy as np
import nibabel as nib
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import anatomy as A  # noqa: E402

HIT_MM = 4.0
ORGANS = ["bladder", "rectum", "prostate", "marrow", "seminal"]


def gate_works(cases, out, models, n4, threads, device="cpu"):
    import seedseg_onnx as SO
    m = SO.SeedSegONNX(models, threads=threads, device=device, log=None)
    rows = []
    for case in cases:
        src = f"{out}/raw/{case}.nii.gz" if n4 else f"{out}/images/{case}.nii.gz"
        d, pm, img = SO.run_nifti(m, src, n4=n4, k=3, log=lambda *a: None)
        rec = json.load(open(f"{out}/params/{case}.json"))
        z = np.array(nib.load(src).header.get_zooms()[:3])
        truth = np.array([mk["center_vox"] for mk in rec["markers"]])
        dets = d["markers"]
        hit, used = 0, set()
        for t in truth:
            dist = [np.linalg.norm((np.array(x["centroid_vox"]) - t) * z) for x in dets]
            j = int(np.argmin(dist)) if dist else -1
            if j >= 0 and dist[j] <= HIT_MM and j not in used:
                hit += 1; used.add(j)
        rows.append(dict(case=case, path="raw+N4" if n4 else "N4-like", markers=len(truth), found=hit,
                         fp=len(dets) - len(used), extra_candidates=len(d["others"]),
                         n_candidates=d["n_candidates"], count_flag=d["count_flag"],
                         min_prob=min([x["mean_prob"] for x in dets], default=0.0),
                         n_calcs=len(rec["calcs"])))
        print(rows[-1], flush=True)
    return rows


def dice(a, b):
    s = a.sum() + b.sum()
    return 2 * (a & b).sum() / s if s else np.nan


def organ_dice(L1, L2):
    ok = (L1 != A.OUTSIDE_FOV) & (L2 != A.OUTSIDE_FOV)
    return np.nanmean([dice((L1 == A.C[o]) & ok, (L2 == A.C[o]) & ok) for o in ORGANS])


def gate_safe(cases, out, anat, subjects):
    real = {s: np.asarray(nib.load(f"{anat}/{s}_lab.nii.gz").dataobj) for s in subjects}
    sids = list(real)
    RR = np.full((len(sids), len(sids)), np.nan)
    for i in range(len(sids)):
        for j in range(i + 1, len(sids)):
            RR[i, j] = RR[j, i] = organ_dice(real[sids[i]], real[sids[j]])
    nn_real = np.nanmax(RR, 1)
    thr = float(np.percentile(nn_real, 95))
    rows = []
    for case in cases:
        L = np.asarray(nib.load(f"{out}/anat1mm/{case}_lab.nii.gz").dataobj)
        d = np.array([organ_dice(L, real[s]) for s in sids])
        rows.append(dict(case=case, nearest_real_dice=float(d.max()), threshold=thr,
                         safe=bool(d.max() < thr), median_real_dice=float(np.median(d))))
    return rows, dict(real_nn_dice_median=float(np.median(nn_real)), real_nn_dice_p95=thr,
                      real_nn_dice_max=float(nn_real.max()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="render.py output dir")
    ap.add_argument("--models", required=True)
    ap.add_argument("--anat", required=True)
    ap.add_argument("--subjects", required=True)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--skip-n4", action="store_true")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--skip-safe", action="store_true")
    ap.add_argument("--skip-works", action="store_true")
    a = ap.parse_args()
    cases = sorted(os.path.basename(p)[:-7] for p in glob.glob(f"{a.out}/images/*.nii.gz"))
    subjects = [l.strip() for l in open(a.subjects) if l.strip()]
    if a.skip_safe:
        rows_s, ref = [dict(case=c, nearest_real_dice=np.nan, safe=False) for c in cases], {}
    else:
        rows_s, ref = gate_safe(cases, a.out, a.anat, subjects)
        pd.DataFrame(rows_s).to_csv(f"{a.out}/gate_safe.csv", index=False)
        print(ref, flush=True)
    if a.skip_works:
        s = pd.DataFrame(rows_s)
        print(s.to_string(index=False))
        return
    rows = gate_works(cases, a.out, a.models, False, a.threads, a.device)
    if not a.skip_n4:
        rows += gate_works(cases, a.out, a.models, True, a.threads, a.device)
    df = pd.DataFrame(rows)
    df.to_csv(f"{a.out}/gate_works.csv", index=False)
    summ = df.groupby("path").agg(cases=("case", "count"), markers=("markers", "sum"), found=("found", "sum"),
                                  fp=("fp", "sum"), extra=("extra_candidates", "sum"), perfect=("found", lambda s: int((s == 3).sum()))).reset_index()
    print(summ.to_string(index=False))
    s = pd.DataFrame(rows_s)
    if ref:
        print(f"safe: {int(s.safe.sum())}/{len(s)}; nearest-real Dice median {s.nearest_real_dice.median():.3f} "
              f"(real-real nearest p95 {ref['real_nn_dice_p95']:.3f})")
    json.dump(dict(works=summ.to_dict("records"), safe_ref=ref, safe_pass=int(s.safe.sum()), n=len(s)),
              open(f"{a.out}/gates.json", "w"), indent=1)


if __name__ == "__main__":
    main()
