"""Locate real gold-marker voids inside (coarse) marker labels and extract patches.

The CT-derived marker labels are ~7 mm blobs; the void centre is found as the
minimum of the lightly smoothed image within the (dilated) label component, refined
to a sub-voxel centroid of the signal deficit. Patches are normalised to the median
CTV intensity.
"""
from __future__ import annotations

import numpy as np
import nibabel as nib
from scipy import ndimage as ndi

ROOT = "/scratch/user/uqaste15/prostate-v2"
HALF = (12, 12, 3)  # patch half-size in voxels (i, j, k)


def load_case(sid, variant="original"):
    im = nib.load(f"{ROOT}/images/{sid}.nii")
    lb = nib.load(f"{ROOT}/labels/{variant}/{sid}.nii.gz")
    return im, np.asarray(im.dataobj).astype(np.float32), np.asarray(lb.dataobj).astype(np.uint8)


def ctv_median(img, lab):
    return float(np.median(img[lab == 2])) if np.any(lab == 2) else float(np.median(img[lab > 0]))


def find_voids(img, lab, ref):
    """Return list of dicts with sub-voxel void centre (voxel coords) and depth."""
    sm = ndi.gaussian_filter(img, (0.7, 0.7, 0.0))
    cc, n = ndi.label(lab == 1)
    out = []
    for c in range(1, n + 1):
        comp = cc == c
        roi = ndi.binary_dilation(comp, iterations=2)
        vals = np.where(roi, sm, np.inf)
        p = np.unravel_index(np.argmin(vals), img.shape)
        # refine: deficit-weighted centroid in a 5x5x3 neighbourhood
        sl = tuple(slice(max(q - h, 0), q + h + 1) for q, h in zip(p, (3, 3, 1)))
        w = np.clip(1.0 - sm[sl] / ref, 0, None) ** 2
        grids = np.meshgrid(*[np.arange(s.start, s.start + w.shape[a]) for a, s in enumerate(sl)], indexing="ij")
        cen = np.array([np.sum(g * w) / np.sum(w) for g in grids]) if w.sum() > 0 else np.array(p, float)
        lab_cen = np.array(ndi.center_of_mass(comp))
        out.append(dict(min_vox=tuple(int(x) for x in p), center=cen.tolist(),
                        depth=float(sm[p] / ref), label_center=lab_cen.tolist(),
                        label_offset_vox=(cen - lab_cen).tolist(), comp_size=int(comp.sum())))
    return out


def extract_patch(vol, center_vox, half=HALF, order=1):
    """Patch centred (sub-voxel, linear interpolation) on center_vox; out-of-bounds -> nan."""
    grids = [np.arange(-h, h + 1) + c for h, c in zip(half, center_vox)]
    # keep slice axis integer (no interpolation through-plane: slices are discrete)
    grids[2] = np.arange(-half[2], half[2] + 1) + int(round(center_vox[2]))
    G = np.meshgrid(*grids, indexing="ij")
    return ndi.map_coordinates(vol, G, order=order, mode="constant", cval=np.nan)


def local_noise_sigma(img, mask):
    """Robust per-channel Gaussian sigma from in-plane high-pass residuals inside mask.

    Uses the difference between a voxel and the mean of its 4 in-plane neighbours
    (Laplacian-type residual, variance 1.25 sigma^2 for white noise) and the MAD.
    Inflated by tissue texture, so treat as an upper bound; see noise calibration in
    validate.py.
    """
    k = np.zeros((3, 3, 1), np.float32)
    k[1, 1, 0] = 1.0
    k[0, 1, 0] = k[2, 1, 0] = k[1, 0, 0] = k[1, 2, 0] = -0.25
    r = ndi.convolve(img, k, mode="nearest")[mask]
    mad = np.median(np.abs(r - np.median(r)))
    return float(1.4826 * mad / np.sqrt(1.25))
