#!/usr/bin/env python3
"""Step 3: population appearance statistics per tissue class.

Per subject, on the 1 mm grid of anatomy.py (intensities r in CTV-median units), for
each class c inside its field-of-view mask eroded by 1 voxel:
  mu      median r
  low     SD of the normalised-convolution low-pass  G5(r m) / G5(m)          (5 mm)
  tex     SD of the band  G1 - G5  (same normalised convolution), with the part
          explained by image noise removed in quadrature (noise leakage factor
          estimated by passing white noise through the same native -> 1 mm sampling)
plus the per-channel noise SD (generator's local_noise_sigma in the dilated CTV,
CTV-median units) on the native image.

Only pooled statistics are written (appearance.json): mean and covariance of
log(mu_c) and log(noise) across subjects, and lognormal fits of low/tex per class.
Per-subject rows go to private/appearance_subjects.csv.
"""
import argparse, json, os, sys
import numpy as np
import nibabel as nib
import pandas as pd
import scipy.ndimage as ndi

import anatomy as A
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "synth"))
import real_voids as RV  # noqa: E402

NC = len(A.CLASSES)


def nconv(x, m, s):
    return ndi.gaussian_filter(x * m, s) / np.maximum(ndi.gaussian_filter(m.astype(np.float32), s), 1e-3)


def leakage_factor(z=(0.6875, 0.6875, 2.0)):
    """SD of the G1-G5 band (1 mm grid) per unit native white-noise SD."""
    rng = np.random.default_rng(0)
    n = rng.standard_normal((160, 160, 30)).astype(np.float32)
    ax = [np.arange(0, s * zz - 1, 1.0) / zz for s, zz in zip(n.shape, z)]
    G = np.meshgrid(*[a + 0.37 for a in ax], indexing="ij")
    r = ndi.map_coordinates(n, G, order=1, mode="nearest")
    m = np.ones(r.shape, bool)
    b = nconv(r, m, 1.0) - nconv(r, m, 5.0)
    return float(b[10:-10, 10:-10, 10:-10].std())


def subject_stats(anat, sid, k_leak):
    lab = np.asarray(nib.load(f"{anat}/{sid}_lab.nii.gz").dataobj)
    r = np.asarray(nib.load(f"{anat}/{sid}_img.nii.gz").dataobj, np.float32)
    im = nib.load(f"{A.ROOT}/images/{sid}.nii")
    vol = np.asarray(im.dataobj, np.float32)
    l2 = np.asarray(nib.load(f"{A.ROOT}/labels/corrected/{sid}.nii.gz").dataobj)
    ref = float(np.median(vol[l2 == 2]))
    sig = RV.local_noise_sigma(vol, ndi.binary_dilation(l2 == 2, iterations=3)) / ref
    row = dict(subject=sid, noise=sig)
    for c, name in enumerate(A.CLASSES):
        m = ndi.binary_erosion(lab == c)
        if m.sum() < 200:
            continue
        low = nconv(r, m, 5.0)
        band = nconv(r, m, 1.0) - low
        row[f"mu_{name}"] = float(np.median(r[m]))
        row[f"low_{name}"] = float(low[m].std())
        row[f"tex_{name}"] = float(np.sqrt(max(band[m].var() - (k_leak * sig) ** 2, 1e-6)))
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anat", required=True)
    ap.add_argument("--subjects", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--nproc", type=int, default=16)
    a = ap.parse_args()
    os.makedirs(f"{a.out}/private", exist_ok=True)
    sids = [l.strip() for l in open(a.subjects) if l.strip()]
    k = leakage_factor()
    from multiprocessing import Pool
    with Pool(a.nproc) as p:
        rows = p.starmap(subject_stats, [(a.anat, s, k) for s in sids])
    df = pd.DataFrame(rows)
    df.to_csv(f"{a.out}/private/appearance_subjects.csv", index=False)

    mu_cols = [f"mu_{n}" for n in A.CLASSES]
    # classes rarely in view (e.g. air) get a fixed floor-level value: impute per subject
    M = np.log(df[mu_cols + ["noise"]].clip(lower=1e-3))
    M = M.fillna(M.median())
    out = dict(classes=A.CLASSES, n_subjects=len(df), noise_leakage=k,
               log_mu_mean=M.mean().tolist(), log_mu_cov=np.cov(M.T.values).tolist(),
               columns=mu_cols + ["noise"])
    for kind in ("low", "tex"):
        L = np.log(df[[f"{kind}_{n}" for n in A.CLASSES]].clip(lower=1e-4))
        out[f"log_{kind}_mean"] = L.mean().fillna(L.mean().median()).tolist()
        out[f"log_{kind}_sd"] = L.std().fillna(L.std().median()).tolist()
    out["summary"] = {c: dict(median=float(df[c].median()), q25=float(df[c].quantile(.25)),
                              q75=float(df[c].quantile(.75)), n=int(df[c].notna().sum()))
                      for c in df.columns if c != "subject"}
    json.dump(out, open(f"{a.out}/appearance.json", "w"), indent=1)
    print(json.dumps(out["summary"], indent=0)[:3000])


if __name__ == "__main__":
    main()
