"""QC figure for generated cases: for every inserted source, original vs synthetic
slice (17 x 17 mm) with the synthetic label contour.

  python scripts/synth/show_case.py /path/synthetic/corrected/params/S002_syn0.json [...] --out fig.png
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402

ROOT = "/scratch/user/uqaste15/prostate-v2"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("params", nargs="+")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = []
    for pj in a.params:
        rec = json.load(open(pj))
        base = os.path.dirname(os.path.dirname(pj))
        orig = np.asarray(nib.load(f"{ROOT}/images/{rec['subject']}.nii").dataobj, np.float32)
        syn = np.asarray(nib.load(f"{base}/images/{rec['case']}.nii.gz").dataobj, np.float32)
        lab = np.asarray(nib.load(f"{base}/labels/{rec['case']}.nii.gz").dataobj)
        ref = rec["noise"]["ctv_median"]
        for s in rec["markers"] + rec["calcs"]:
            c = np.rint(s["center_vox"]).astype(int)
            if s["type"] == "marker":
                desc = f"marker th={s['theta_deg']:.0f} dchi={s['dchi_ppm']:.0f} {s['context']}"
            else:
                desc = f"{'para' if s['paramagnetic'] else 'dia'} d={s['equiv_diam_mm']:.1f} dchi={s['dchi_ppm']:+.1f} rho={s['rho']:.1f}"
            rows.append((rec["case"], desc, c, orig, syn, lab, ref))
    n = len(rows)
    fig, ax = plt.subplots(n, 4, figsize=(6.4, 1.7 * n))
    ax = np.atleast_2d(ax)
    for r, (case, desc, c, orig, syn, lab, ref) in enumerate(rows):
        sl = (slice(c[0] - 12, c[0] + 13), slice(c[1] - 12, c[1] + 13))
        for j, (vol, dk, t) in enumerate([(orig, 0, "original"), (syn, -1, "syn k-1"), (syn, 0, "syn k"), (syn, 1, "syn k+1")]):
            k = c[2] + dk
            ax[r, j].imshow(vol[sl + (k,)].T / ref, cmap="gray", vmin=0, vmax=1.6, origin="lower")
            if j > 0:
                ax[r, j].contour((lab[sl + (k,)] == 1).T, levels=[0.5], colors=["#eb6834"], linewidths=0.8)
            ax[r, j].set_xticks([]); ax[r, j].set_yticks([])
            if r == 0:
                ax[r, j].set_title(t, fontsize=8)
        ax[r, 0].set_ylabel(f"{case}\n{desc}", fontsize=5.5)
    fig.tight_layout()
    fig.savefig(a.out, dpi=150)


if __name__ == "__main__":
    main()
