"""Numerical checks of the forward model (results go in the methods/supplement).

1. Dipole convolution vs the analytic field of a uniformly magnetised sphere
   (dB/B0 = dchi/3 (a/r)^3 (3 cos^2 th - 1) outside), as a function of padding.
2. Convergence of the voxel attenuation |A| with fine-grid oversampling (in-plane and
   through-plane) and padding, for a gold cylinder at 0, 45 and 90 deg to B0.
Writes physics_checks.json to the validation directory.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import physics as P  # noqa: E402

OUT = "/scratch/user/uqaste15/prostate-v2/synthetic/validation"


def sphere_check():
    h, n, a, dchi = 0.1, 200, 1.5, -25.0
    c = (np.arange(n) - (n - 1) / 2) * h
    f, ext = P.ellipsoid_union_inside([0, 0, 0], [([0, 0, 0], [a, a, a], np.eye(3))])
    occ = P.rasterise(f, ext, np.zeros(3), [c, c, c], 3)
    X, Y, Z = np.meshgrid(c, c, c, indexing="ij")
    r = np.sqrt(X ** 2 + Y ** 2 + Z ** 2)
    m = (r > 1.5 * a) & (r < 8)
    ana = dchi / 3 * (a / r[m]) ** 3 * (3 * (Z[m] / r[m]) ** 2 - 1)
    out = {}
    for pad in (1.0, 1.5, 2.0):
        fld = P.dipole_field(occ * dchi, [h] * 3, (0, 0, 1), pad=pad)[m]
        err = np.abs(fld - ana)
        out[f"pad{pad}"] = dict(max_abs_err_ppm=float(err.max()),
                                max_abs_err_Hz_3T=float(err.max() * 1e-6 * P.GAMMA_BAR * 3.0),
                                rms_err_ppm=float(np.sqrt(np.mean(err ** 2))),
                                max_field_ppm=float(np.abs(ana).max()))
    return out


def convergence():
    acq = P.Acquisition()
    n = (25, 25, 9)
    vox = np.array(acq.voxel)
    res = {}
    rng = np.random.default_rng(0)
    offs = rng.uniform(-0.5, 0.5, (4, 3)) * vox
    ref_grid = P.Grid(os_xy=9, os_z=20, ss=3, pad=2.0)
    for th in (0, 45, 90):
        u = [np.sin(np.radians(th)), 0, np.cos(np.radians(th))]
        for gname, g in [("os5_z10_pad1.5", P.Grid(os_xy=5, os_z=10)), ("os7_z14_pad1.5", P.Grid()),
                         ("os7_z14_pad2", P.Grid(pad=2.0)), ("os3_z6_pad1.5", P.Grid(os_xy=3, os_z=6))]:
            errs, p99, t = [], [], 0.0
            for o in offs:
                cen = (np.array(n) // 2) * vox + o
                src = [P.gold_marker(u)(cen)]
                Aref = np.abs(P.simulate_patch(src, cen, n, acq, ref_grid))
                t0 = time.time()
                A = np.abs(P.simulate_patch(src, cen, n, acq, g))
                t += time.time() - t0
                e = np.abs(A - Aref)
                errs.append(e.max())
                p99.append(np.percentile(e[Aref < 0.99], 99))
            res[f"theta{th}_{gname}"] = dict(max_abs_err_vs_os9_z20=float(np.max(errs)),
                                             p99_abs_err_in_void=float(np.max(p99)),
                                             time_per_patch_s=t / len(offs))
    return res


if __name__ == "__main__":
    out = dict(sphere=sphere_check(), convergence=convergence())
    os.makedirs(OUT, exist_ok=True)
    with open(f"{OUT}/physics_checks.json", "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))
