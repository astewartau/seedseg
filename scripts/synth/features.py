"""Rotation-invariant descriptors of a void patch (normalised to CTV median).

Patch layout: (25, 25, 7) voxels, in-plane centred (sub-voxel) on the void, slice
index 3 = slice nearest the void centre. Descriptors are re-aligned on the slice of
minimum intensity so that real and simulated voids are compared identically.
"""
from __future__ import annotations

import numpy as np

VOX = (0.6875, 0.6875, 2.0)
RMAX = 9  # radial bins (pixels)
NSL = 5   # slices kept around the minimum slice
THR = 0.6  # void threshold (fraction of CTV median)


def _radial_index(n=25):
    c = (n - 1) / 2
    y, x = np.mgrid[:n, :n]
    return np.rint(np.hypot(x - c, y - c)).astype(int)


_RIDX = _radial_index()


def describe(patch):
    p = np.nan_to_num(patch, nan=1.0)
    cen = p[9:16, 9:16, :]
    mins = cen.min(axis=(0, 1))
    k0 = int(np.argmin(mins[1:-1]) + 1)  # restrict to inner slices
    ks = [k0 + d for d in range(-(NSL // 2), NSL // 2 + 1)]
    sl = np.stack([p[:, :, k] if 0 <= k < p.shape[2] else np.ones(p.shape[:2]) for k in ks], -1)
    radial = np.zeros((RMAX, NSL))
    for r in range(RMAX):
        m = _RIDX == r
        radial[r] = sl[m].mean(axis=0)
    # void area per slice: connected-ish region within 6 px of centre below THR
    disk = _RIDX <= 6
    area = ((sl < THR) & disk[..., None]).sum(axis=(0, 1)) * VOX[0] * VOX[1]  # mm^2
    # in-plane elongation on the minimum slice from deficit second moments
    d = np.clip(1.0 - sl[:, :, NSL // 2], 0, None) * disk
    y, x = np.mgrid[:25, :25]
    w = d.sum()
    if w > 0:
        mx, my = (d * x).sum() / w, (d * y).sum() / w
        cxx = (d * (x - mx) ** 2).sum() / w
        cyy = (d * (y - my) ** 2).sum() / w
        cxy = (d * (x - mx) * (y - my)).sum() / w
        ev = np.linalg.eigvalsh([[cxx, cxy], [cxy, cyy]])
        elong = float(np.sqrt(max(ev[1], 1e-9) / max(ev[0], 1e-9)))
    else:
        elong = 1.0
    nslices = int((cen.min(axis=(0, 1)) < THR).sum())
    return dict(radial=radial, mins=sl[9:16, 9:16].min(axis=(0, 1)), area=area,
                elong=elong, nslices=nslices, min=float(mins.min()),
                vol=float(area.sum() * VOX[2]))


def vector(desc):
    """Feature vector used for nearest-neighbour matching."""
    return np.concatenate([desc["radial"].ravel(), [0.1 * desc["elong"]]])
