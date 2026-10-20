"""Validation figures: real vs simulated gold-marker voids.

Reads metrics_{variant}.json / patches_{variant}.npz written by validate.py and writes
  fig_montage_{variant}.png   real voids next to simulated voids (3 slices each)
  fig_profiles_{variant}.png  radial profiles, per-slice minimum/area, elongation, volume
  fig_fit_{variant}.png       fitted posterior over polar angle and delta-chi
  fig_meanvoid_{variant}.png  population-mean void, real vs simulated
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import features as F  # noqa: E402

C_REAL, C_SIM = "#2a78d6", "#eb6834"
INK, MUTED = "#222222", "#888888"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e6e6",
                     "grid.linewidth": 0.6, "legend.frameon": False})


def min_slice(p):
    c = np.nan_to_num(p, nan=1.0)[9:16, 9:16]
    return int(np.argmin(c.min(axis=(0, 1))[1:-1]) + 1)


def montage(real, sim, out, n=12, seed=0, title=""):
    rng = np.random.default_rng(seed)
    ir = rng.choice(len(real), n, replace=False)
    isim = rng.choice(len(sim), n, replace=False)
    fig, ax = plt.subplots(n // 2, 12, figsize=(12, n / 2 * 1.08))
    for row in range(n // 2):
        for half, (arr, idx, lab) in enumerate([(real, ir, "real"), (sim, isim, "sim")]):
            for j in range(2):
                p = arr[idx[row * 2 + j]]
                k = min_slice(p)
                for d in range(3):
                    a = ax[row, half * 6 + j * 3 + d]
                    kk = int(np.clip(k + d - 1, 0, p.shape[2] - 1))
                    a.imshow(np.nan_to_num(p[:, :, kk], nan=1).T, cmap="gray", vmin=0, vmax=1.5,
                             origin="lower", interpolation="nearest")
                    a.set_xticks([]); a.set_yticks([])
                    for s in a.spines.values():
                        s.set_visible(False)
                    if row == 0 and d == 1:
                        a.set_title(f"{lab}  (slices -1, 0, +1)", fontsize=8,
                                    color=C_REAL if lab == "real" else C_SIM)
    fig.suptitle(title or "Real (left 6 columns) vs simulated (right 6 columns) voids; "
                 "17 x 17 mm, intensity / CTV median, window 0-1.5", fontsize=9)
    plt.subplots_adjust(wspace=0.04, hspace=0.06, left=0.01, right=0.99, top=0.93, bottom=0.01)
    fig.savefig(out, dpi=150)
    plt.close(fig)


def band(ax, x, arr, color, label):
    q = np.nanpercentile(arr, [25, 50, 75], axis=0)
    ax.fill_between(x, q[0], q[2], color=color, alpha=0.18, linewidth=0)
    ax.plot(x, q[1], color=color, lw=2, label=label)


def profiles(rd, sd, out, nr, ns):
    fig, ax = plt.subplots(2, 3, figsize=(11, 6.2))
    r_mm = np.arange(F.RMAX) * F.VOX[0]
    for i, (dk, name) in enumerate([(0, "minimum slice"), (1, "adjacent slices (mean of +-1)")]):
        a = ax[0, i]
        if dk == 0:
            R_ = np.array([d["radial"][:, F.NSL // 2] for d in rd])
            S_ = np.array([d["radial"][:, F.NSL // 2] for d in sd])
        else:
            R_ = np.array([(d["radial"][:, 1] + d["radial"][:, 3]) / 2 for d in rd])
            S_ = np.array([(d["radial"][:, 1] + d["radial"][:, 3]) / 2 for d in sd])
        band(a, r_mm, R_, C_REAL, f"real (n={nr})")
        band(a, r_mm, S_, C_SIM, f"simulated (n={ns})")
        a.set_xlabel("radius from void centre (mm)"); a.set_ylabel("intensity / CTV median")
        a.set_title(f"Radial profile, {name}", fontsize=9, loc="left"); a.set_ylim(0, 1.15)
        a.legend(loc="lower right")
    zs = (np.arange(F.NSL) - F.NSL // 2) * F.VOX[2]
    a = ax[0, 2]
    band(a, zs, np.array([d["mins"] for d in rd]), C_REAL, "real")
    band(a, zs, np.array([d["mins"] for d in sd]), C_SIM, "simulated")
    a.set_xlabel("slice offset from minimum slice (mm)"); a.set_ylabel("min intensity / CTV median")
    a.set_title("Per-slice minimum (central 4.8 mm)", fontsize=9, loc="left"); a.legend(loc="lower right")
    a = ax[1, 0]
    band(a, zs, np.array([d["area"] for d in rd]), C_REAL, "real")
    band(a, zs, np.array([d["area"] for d in sd]), C_SIM, "simulated")
    a.set_xlabel("slice offset from minimum slice (mm)"); a.set_ylabel(f"area < {F.THR} (mm$^2$)")
    a.set_title("Void area per slice", fontsize=9, loc="left"); a.legend(loc="upper right")
    for a, key, lab, bins in [(ax[1, 1], "elong", "in-plane elongation (sqrt eigenvalue ratio)", np.linspace(1, 2.5, 25)),
                              (ax[1, 2], "vol", f"void volume < {F.THR} (mm$^3$)", np.linspace(0, 140, 29))]:
        r = [d[key] for d in rd]
        s = [d[key] for d in sd]
        a.hist(r, bins=bins, density=True, histtype="step", lw=2, color=C_REAL, label="real")
        a.hist(s, bins=bins, density=True, histtype="step", lw=2, color=C_SIM, label="simulated")
        a.set_xlabel(lab); a.set_ylabel("density"); a.legend()
    ax[1, 1].set_title("Elongation (minimum slice)", fontsize=9, loc="left")
    ax[1, 2].set_title("Void volume", fontsize=9, loc="left")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fit_fig(m, z, out):
    pp = np.array(m["pooled_posterior"]["p"])
    th = np.array(m["pooled_posterior"]["thetas"])
    dc = np.array(m["pooled_posterior"]["dchis"])
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.2))
    iso = np.sin(np.radians(th + 5)) - np.sin(np.radians(np.maximum(th - 5, 0)))
    iso = np.cos(np.radians(np.maximum(th - 5, 0))) - np.cos(np.radians(np.minimum(th + 5, 90)))
    iso /= iso.sum()
    w = 4
    ax[0].bar(th - w / 2, pp.sum(1), width=w, color=C_REAL, label="fitted (pooled posterior)")
    ax[0].bar(th + w / 2, iso, width=w, color=MUTED, label="isotropic prior (sin theta)")
    ax[0].set_xlabel("polar angle to B0 (deg)"); ax[0].set_ylabel("probability"); ax[0].legend(fontsize=7)
    ax[0].set_title("Orientation", fontsize=9, loc="left")
    ax[1].bar(dc, pp.sum(0), width=4, color=C_REAL)
    ax[1].set_xlabel("delta-chi (ppm)"); ax[1].set_ylabel("probability")
    ax[1].set_title("Susceptibility difference", fontsize=9, loc="left")
    im = ax[2].imshow(pp.T, aspect="auto", origin="lower", cmap="Blues",
                      extent=[th[0] - 5, th[-1] + 5, -0.5, len(dc) - 0.5])
    ax[2].set_yticks(range(len(dc))); ax[2].set_yticklabels([f"{d:g}" for d in dc])
    ax[2].set_xlabel("polar angle (deg)"); ax[2].set_ylabel("delta-chi (ppm)"); ax[2].grid(False)
    ax[2].set_title("Joint posterior", fontsize=9, loc="left")
    fig.colorbar(im, ax=ax[2])
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def align(p):
    k = min_slice(p)
    return np.stack([np.nan_to_num(p[:, :, int(np.clip(k + d, 0, p.shape[2] - 1))], nan=1) for d in (-1, 0, 1)], -1)


def meanvoid(real, sim, out):
    mr = np.mean([align(p) for p in real], 0)
    ms = np.mean([align(p) for p in sim], 0)
    fig, ax = plt.subplots(3, 3, figsize=(6.4, 6.6))
    for j in range(3):
        for i, (arr, lab) in enumerate([(mr, "real mean"), (ms, "simulated mean")]):
            ax[i, j].imshow(arr[:, :, j].T, cmap="gray", vmin=0.2, vmax=1.2, origin="lower")
            ax[i, j].set_title(f"{lab}, slice {j - 1:+d}", fontsize=8)
        im = ax[2, j].imshow((ms - mr)[:, :, j].T, cmap="RdBu_r", vmin=-0.2, vmax=0.2, origin="lower")
        ax[2, j].set_title(f"sim - real, slice {j - 1:+d}", fontsize=8)
    for a in ax.ravel():
        a.set_xticks([]); a.set_yticks([]); a.grid(False)
    fig.colorbar(im, ax=ax[2, :].tolist(), shrink=0.8, label="difference / CTV median")
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="base")
    ap.add_argument("--dir", default="/scratch/user/uqaste15/prostate-v2/synthetic/validation")
    a = ap.parse_args()
    m = json.load(open(f"{a.dir}/metrics_{a.variant}.json"))
    z = np.load(f"{a.dir}/patches_{a.variant}.npz")
    real, sim = z["real"], z["matched"]
    rd = [F.describe(p) for p in real]
    sd = [F.describe(p) for p in sim]
    montage(real, sim, f"{a.dir}/fig_montage_{a.variant}.png")
    profiles(rd, sd, f"{a.dir}/fig_profiles_{a.variant}.png", len(real), len(sim))
    fit_fig(m, z, f"{a.dir}/fig_fit_{a.variant}.png")
    meanvoid(real, sim, f"{a.dir}/fig_meanvoid_{a.variant}.png")
    print("figures written to", a.dir)


if __name__ == "__main__":
    main()
