"""Calibrate the magnitude noise model from signal-free regions (rectal/bowel gas).

For every subject the largest connected region near the prostate whose 5x5 in-plane
mean is < 0.6 x CTV median (eroded by 2 px) is taken as signal-free. In such a region a
sum-of-squares magnitude over N effective channels follows a scaled chi distribution
with 2N degrees of freedom: mean = sigma * sqrt(2) Gamma(N+1/2)/Gamma(N),
var = sigma^2 (2N - mean^2/sigma^2). sigma (per-channel SD) is estimated independently
from tissue (high-pass MAD inside the CTV, real_voids.local_noise_sigma). N_eff is
solved from mean/sigma. Lag-1 noise autocorrelations are measured inside the gas region.

Writes noise_calibration.csv / .json to the validation directory.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy.optimize import brentq
from scipy.special import gammaln

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import validate as V  # noqa: E402

OUT = "/scratch/user/uqaste15/prostate-v2/synthetic/validation"


def chi_mean_ratio(n):
    return np.sqrt(2.0) * np.exp(gammaln(n + 0.5) - gammaln(n))


def n_eff_from_ratio(r):
    if r <= chi_mean_ratio(1.0):
        return 1.0
    return brentq(lambda n: chi_mean_ratio(n) - r, 1.0, 200.0)


def gas_region(img_n, lab):
    sm = ndi.uniform_filter(img_n, (5, 5, 1))
    pc = np.array(ndi.center_of_mass(lab == 2)).astype(int)
    m = np.zeros(img_n.shape, bool)
    m[max(pc[0] - 60, 0):pc[0] + 60, max(pc[1] - 60, 0):pc[1] + 60, max(pc[2] - 15, 0):pc[2] + 15] = True
    dark = (sm < 0.6) & m & (img_n > 0)
    dark = ndi.binary_erosion(dark, structure=np.ones((5, 5, 1)))
    cc, n = ndi.label(dark)
    if n == 0:
        return None
    sizes = ndi.sum(dark, cc, range(1, n + 1))
    k = int(np.argmax(sizes)) + 1
    return cc == k if sizes[k - 1] >= 150 else None


def main():
    rows = []
    for sid in V.reference_subjects():
        s = V.subject(sid)
        img_n = s["img"] / s["ref"]
        reg = gas_region(img_n, s["lab"])
        if reg is None:
            continue
        v = img_n[reg]
        sig = s["sigma"] / s["ref"]
        d = img_n - img_n[reg].mean()
        corr = []
        for ax in range(3):
            both = reg & np.roll(reg, 1, ax)
            corr.append(float(np.sum(d[both] * np.roll(d, 1, ax)[both]) / np.sum(d[both] ** 2)))
        rows.append(dict(sid=sid, n_vox=int(reg.sum()), floor_mean=float(v.mean()), floor_median=float(np.median(v)),
                         floor_sd=float(v.std()), sigma_tissue=float(sig), ratio=float(v.mean() / sig),
                         sd_ratio=float(v.std() / sig), corr_i=corr[0], corr_j=corr[1], corr_k=corr[2]))
    df = pd.DataFrame(rows)
    # regions with the lowest floor are least contaminated by residual signal
    q = df.ratio.quantile(0.25)
    r_lo = float(df.ratio[df.ratio <= df.ratio.median()].median())
    summary = dict(n_subjects=len(df), ratio_median=float(df.ratio.median()), ratio_q25=float(q),
                   n_eff_from_median=n_eff_from_ratio(df.ratio.median()),
                   n_eff_from_q25=n_eff_from_ratio(q), n_eff_from_lower_half=n_eff_from_ratio(r_lo),
                   floor_sd_ratio_median=float(df.sd_ratio.median()),
                   corr_median=[float(df[c].median()) for c in ("corr_i", "corr_j", "corr_k")],
                   sigma_tissue_median=float(df.sigma_tissue.median()))
    os.makedirs(OUT, exist_ok=True)
    df.to_csv(f"{OUT}/noise_calibration.csv", index=False)
    with open(f"{OUT}/noise_calibration.json", "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
