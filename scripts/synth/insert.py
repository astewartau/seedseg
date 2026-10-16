"""Insert simulated susceptibility sources into a real magnitude volume.

A *source spec* is a JSON-serialisable dict:

  marker: {"type": "marker", "center_vox": [i, j, k], "axis": [ui, uj, uk] (voxel-axis
           mm frame), "dchi_ppm": -25.0, "radius_mm": 0.5, "length_mm": 3.0}
  calc:   {"type": "calc", "center_vox": [...], "dchi_ppm": -4.0, "rho": 0.2,
           "ellipsoids": [{"offset_mm": [...], "semi_axes_mm": [...], "rot": 3x3}, ...]}

Each source is simulated in its own MR patch (default 25 x 25 x 9 voxels = 17 x 17 x
18 mm); complex attenuations of different sources are multiplied (exact when the
sources' perturbations do not overlap, which the placement rules ensure).
"""
from __future__ import annotations

import numpy as np

from physics import (Acquisition, Grid, cylinder_inside, ellipsoid_union_inside,
                     simulate_patch, occupancy_on_mr)

PATCH_HALF = (12, 12, 4)


def spec_to_source(spec, center_mm):
    if spec["type"] == "marker":
        f, ext = cylinder_inside(center_mm, spec["axis"], spec.get("radius_mm", 0.5), spec.get("length_mm", 3.0))
        return dict(inside=f, extent=ext, dchi=spec["dchi_ppm"], rho=0.0, center=center_mm)
    if spec["type"] == "calc":
        ells = [(e["offset_mm"], e["semi_axes_mm"], e["rot"]) for e in spec["ellipsoids"]]
        f, ext = ellipsoid_union_inside(center_mm, ells)
        return dict(inside=f, extent=ext, dchi=spec["dchi_ppm"], rho=spec.get("rho", 0.0), center=center_mm)
    raise ValueError(spec["type"])


def patch_bounds(center_vox, shape, half=PATCH_HALF):
    ic = np.rint(center_vox).astype(int)
    lo = ic - np.array(half)
    hi = ic + np.array(half) + 1
    if np.any(lo < 0) or np.any(hi > np.array(shape)):
        return None
    return lo, hi


def simulate_spec(spec, shape, acq: Acquisition, grid: Grid, half=PATCH_HALF):
    """Return (slices, A_patch, occ_mr_patch) for one spec, or None if out of bounds."""
    c = np.asarray(spec["center_vox"], float)
    b = patch_bounds(c, shape, half)
    if b is None:
        return None
    lo, hi = b
    n = tuple(int(x) for x in hi - lo)
    vox = np.asarray(acq.voxel, float)
    center_mm = (c - lo) * vox
    src = spec_to_source(spec, center_mm)
    A, fine = simulate_patch([src], center_mm, n, acq, grid, return_fine=True)
    occ = occupancy_on_mr(fine["occ"], n, acq, grid)
    sl = tuple(slice(int(a), int(b_)) for a, b_ in zip(lo, hi))
    return sl, A, occ


def correlated_noise(shape, rng, corr=(0.0, 0.0)):
    """Complex white Gaussian noise (unit variance per channel) with lag-1 correlation
    corr[a] along axis a (a = 0, 1), via separable 3-tap kernels [c, 1, c]/norm."""
    n = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    if np.isscalar(corr):
        corr = (corr, 0.0)
    for ax, r in enumerate(corr):
        if r > 0:
            # rho = 2c / (1 + 2c^2)  ->  c = (1 - sqrt(1 - 2 rho^2)) / (2 rho)
            r = min(r, 0.7)
            c = (1 - np.sqrt(1 - 2 * r * r)) / (2 * r)
            n = (n + c * np.roll(n, 1, ax) + c * np.roll(n, -1, ax)) / np.sqrt(1 + 2 * c * c)
    return n


def insert_sources(img, specs, acq: Acquisition, grid: Grid, sigma, rng,
                   noise_corr=(0.38, 0.22), n_coils=10, half=PATCH_HALF):
    """Return (new_img, A_total, per_spec_info).

    Noise model (multi-channel sum-of-squares magnitude, non-central chi with 2N dof):
      M'^2 = |A|^2 M^2 + (1 - |A|^2)_+ * sigma^2 * sum_{c=1..N} |n_c|^2,  n_c ~ CN(0, 1)
    so that E[M'^2] = |A|^2 S^2 + 2 N sigma^2 (the exact expectation for the attenuated
    signal), the image is unchanged where A = 1 and the void floor has the scanner's
    noise-floor distribution. n_coils = 1 gives the Rician (single-channel) case.
    sigma : scalar or array broadcastable to img -- per-channel Gaussian noise SD of img
            (equals the SD of the magnitude in tissue at high SNR).
    """
    A_tot = np.ones(img.shape, np.complex64)
    infos = []
    for spec in specs:
        r = simulate_spec(spec, img.shape, acq, grid, half)
        if r is None:
            infos.append(None)
            continue
        sl, A, occ = r
        A_tot[sl] *= A
        infos.append(dict(slices=sl, A=A, occ=occ))
    out = img.astype(np.float32).copy()
    touched = np.abs(A_tot - 1) > 1e-5
    if touched.any():
        idx = np.nonzero(touched)
        bb = tuple(slice(i.min(), i.max() + 1) for i in idx)
        a2 = np.abs(A_tot[bb]) ** 2
        s = sigma[bb] if np.ndim(sigma) else sigma
        floor = np.zeros(a2.shape)
        for _ in range(int(n_coils)):
            floor += np.abs(correlated_noise(a2.shape, rng, noise_corr)) ** 2
        m2 = a2 * img[bb].astype(np.float64) ** 2 + np.clip(1.0 - a2, 0, None) * s ** 2 * floor
        out[bb] = np.where(touched[bb], np.sqrt(m2), img[bb])
    return out, A_tot, infos
