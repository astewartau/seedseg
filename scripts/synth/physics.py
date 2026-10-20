"""Forward model for susceptibility-induced signal voids in 2D multi-slice GRE magnitude.

Coordinates
-----------
All geometry is expressed in millimetres in the *voxel-axis frame* of the MR image
(axis 0 = i, axis 1 = j, axis 2 = k = slice direction). For the prostate data the
affine is a pure axial acquisition (HFS) so the slice axis k is parallel to the
scanner z axis, i.e. B0. The B0 direction is nevertheless passed explicitly (unit
vector in the voxel-axis frame, computed from the affine) so the model also holds
for oblique acquisitions.

Model (see README.md for the equations)
---------------------------------------
1. A source (gold cylinder or calcification) is rasterised as a fractional-occupancy
   map ``occ`` on a fine grid that over-samples each MR voxel ``os_xy`` x ``os_xy`` x
   ``os_z`` times (super-sampled ``ss``^3 per fine cell for partial volume).
2. Delta-chi map = occ * dchi (ppm, relative to tissue). Field perturbation
   dB/B0 = IFFT[ D(k) FFT[dchi] ], D(k) = 1/3 - (k.b0)^2/|k|^2, on a zero-padded grid.
3. Off-resonance df = gamma_bar * B0 * dB/B0 [Hz]; phase at TE: phi = 2 pi df TE.
4. Each fine cell carries transverse magnetisation m = rho * exp(-i phi), rho = 1 - occ
   + occ * rho_source (rho_source = 0 for gold).
5. Slice selection: the fine cell at through-plane position z contributes to slice s
   with weight p(z + dz_ss - z_s), where p is the slice profile (boxcar or Gaussian of
   thickness/FWHM = slice thickness) and dz_ss = sign_ss * df / G_ss is the optional
   off-resonance slice displacement (G_ss = RF bandwidth / slice thickness).
6. Readout: the cell is deposited at in-plane readout coordinate x + dx_ro with
   dx_ro = sign_ro * df / BW_px (pixels) (optional).
7. In-plane point-spread: either boxcar average over the MR pixel ('box') or ideal
   k-space truncation to the acquisition matrix followed by sampling at pixel centres
   ('sinc').
8. Complex attenuation A = S / S0 where S0 is the same computation with no source.
   The synthetic magnitude is |A * I + n|, n ~ CN(0, sigma^2 max(0, 1 - |A|^2)) so the
   noise level of the final image equals that of the original image (Rician floor in the
   void is preserved).
"""
from __future__ import annotations

import dataclasses
import math
from typing import Optional

import numpy as np
import scipy.fft as sfft

GAMMA_BAR = 42.577478518e6  # Hz/T, 1H


@dataclasses.dataclass
class Acquisition:
    """Sequence and reconstruction parameters (defaults: prostate T1w 2D FLASH)."""

    B0: float = 3.0                  # T
    TE: float = 6.66e-3              # s
    voxel: tuple = (0.6875, 0.6875, 2.0)  # mm (i, j, k)
    b0_dir: tuple = (0.0, 0.0, 1.0)  # unit vector in voxel-axis frame
    slice_axis: int = 2
    slice_profile: str = "box"       # 'box' | 'gauss'
    slice_fwhm: Optional[float] = None  # mm, defaults to voxel[slice_axis]
    readout_axis: Optional[int] = None  # 0 or 1; None disables readout shift
    bw_px: float = 250.0             # Hz/pixel readout bandwidth
    sign_ro: float = 1.0
    rf_bw: Optional[float] = None    # Hz, RF pulse bandwidth; None disables slice shift
    sign_ss: float = 1.0
    psf: str = "box"                 # 'box' | 'sinc'
    acq_frac: tuple = (1.0, 1.0)     # acquired / reconstructed matrix (in-plane), for 'sinc'
    blur_fwhm: float = 0.0           # mm, extra in-plane Gaussian PSF (filters, resampling, motion)
    blur_fwhm_z: float = 0.0         # mm, extra through-plane Gaussian PSF (applied across slices)

    def to_dict(self):
        return dataclasses.asdict(self)


@dataclasses.dataclass
class Grid:
    """Fine-grid sampling of an MR patch."""

    os_xy: int = 7
    os_z: int = 14
    ss: int = 3          # super-sampling per fine cell for occupancy
    pad: float = 1.5     # zero-padding factor for the dipole convolution


# --------------------------------------------------------------------------
# Source geometry (occupancy functions take points (N,3) in mm, return bool)
# --------------------------------------------------------------------------

def cylinder_inside(center, axis, radius, length):
    c = np.asarray(center, float)
    u = np.asarray(axis, float)
    u = u / np.linalg.norm(u)

    def f(p):
        d = p - c
        t = d @ u
        rad2 = np.einsum("ij,ij->i", d, d) - t * t
        return (np.abs(t) <= 0.5 * length) & (rad2 <= radius * radius)

    extent = math.hypot(0.5 * length, radius)
    return f, extent


def ellipsoid_union_inside(center, ellipsoids):
    """ellipsoids: list of (offset(3), semi_axes(3), rotation(3x3))."""
    c = np.asarray(center, float)
    ells = [(np.asarray(o, float), np.asarray(a, float), np.asarray(R, float)) for o, a, R in ellipsoids]

    def f(p):
        out = np.zeros(len(p), bool)
        for o, a, R in ells:
            q = (p - c - o) @ R  # coordinates in ellipsoid frame (R columns = axes)
            out |= np.sum((q / a) ** 2, axis=1) <= 1.0
        return out

    extent = max(np.linalg.norm(o) + a.max() for o, a, _ in ells)
    return f, extent


# --------------------------------------------------------------------------
# Patch simulation
# --------------------------------------------------------------------------

def _fine_coords(n_mr, os, vox):
    """Fine-cell centre coordinates (mm) for n_mr MR voxels, voxel 0 centred at 0."""
    idx = np.arange(n_mr * os)
    return ((idx + 0.5) / os - 0.5) * vox


def rasterise(inside, extent, center, coords, ss):
    """Fractional occupancy on the fine grid given by coords=(x,y,z) 1-D arrays."""
    shape = tuple(len(c) for c in coords)
    occ = np.zeros(shape, np.float32)
    sl = []
    for ax, c in enumerate(coords):
        h = c[1] - c[0]
        lo = np.searchsorted(c, center[ax] - extent - h)
        hi = np.searchsorted(c, center[ax] + extent + h)
        sl.append(slice(max(lo, 0), min(hi, len(c))))
    sub = [c[s] for c, s in zip(coords, sl)]
    if any(len(s) == 0 for s in sub):
        return occ
    hs = [c[1] - c[0] for c in coords]
    offs = [((np.arange(ss) + 0.5) / ss - 0.5) * h for h in hs]
    X, Y, Z = np.meshgrid(*sub, indexing="ij")
    base = np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1)
    acc = np.zeros(len(base), np.float32)
    for ox in offs[0]:
        for oy in offs[1]:
            for oz in offs[2]:
                acc += inside(base + np.array([ox, oy, oz]))
    occ[tuple(sl)] = (acc / ss ** 3).reshape(X.shape)
    return occ


def dipole_field(dchi, h, b0_dir, pad=1.5, workers=-1):
    """dB/B0 (same units as dchi) via the k-space dipole kernel on a zero-padded grid."""
    shape = np.array(dchi.shape)
    pshape = [sfft.next_fast_len(int(math.ceil(n * pad))) for n in shape]
    ks = [sfft.fftfreq(n, d=hh) for n, hh in zip(pshape, h)]
    KX, KY, KZ = np.meshgrid(*ks, indexing="ij", sparse=True)
    b = np.asarray(b0_dir, float)
    b = b / np.linalg.norm(b)
    kb = KX * b[0] + KY * b[1] + KZ * b[2]
    k2 = KX ** 2 + KY ** 2 + KZ ** 2
    with np.errstate(invalid="ignore", divide="ignore"):
        D = (1.0 / 3.0 - kb ** 2 / k2).astype(np.float32)
    D[0, 0, 0] = 0.0
    buf = np.zeros(pshape, np.float32)
    buf[: shape[0], : shape[1], : shape[2]] = dchi
    F = sfft.rfftn(buf, workers=workers)
    D = D[:, :, : F.shape[2]]
    field = sfft.irfftn(F * D, s=pshape, workers=workers)
    return field[: shape[0], : shape[1], : shape[2]].astype(np.float32)


def _slice_profile(dz, thick, kind):
    if kind == "box":
        return (np.abs(dz) <= 0.5 * thick).astype(np.float32)
    if kind == "gauss":
        s = thick / 2.3548
        return np.exp(-0.5 * (dz / s) ** 2).astype(np.float32)
    raise ValueError(kind)


def simulate_patch(sources, center_mm, n_mr, acq: Acquisition, grid: Grid = Grid(),
                   return_fine=False):
    """Complex attenuation A on an MR patch.

    sources : list of dicts with keys 'inside', 'extent', 'dchi' (ppm, relative),
              'rho' (relative proton density inside the source, 0 for gold).
    center_mm : (3,) source centre in patch mm coordinates (voxel 0 centre = 0).
    n_mr : (ni, nj, nk) MR voxels in the patch.
    """
    vox = np.asarray(acq.voxel, float)
    sa = acq.slice_axis
    os = [grid.os_xy, grid.os_xy, grid.os_xy]
    os[sa] = grid.os_z
    coords = [_fine_coords(n, o, v) for n, o, v in zip(n_mr, os, vox)]
    h = [c[1] - c[0] for c in coords]

    occ_tot = np.zeros([len(c) for c in coords], np.float32)
    dchi = np.zeros_like(occ_tot)
    rho = np.ones_like(occ_tot)
    for s in sources:
        occ = rasterise(s["inside"], s["extent"], s.get("center", center_mm), coords, grid.ss)
        dchi += occ * s["dchi"]
        rho -= occ * (1.0 - s["rho"])
        occ_tot += occ
    rho = np.clip(rho, 0, None)

    field_ppm = dipole_field(dchi, h, acq.b0_dir, pad=grid.pad)
    df = GAMMA_BAR * acq.B0 * field_ppm * 1e-6  # Hz
    m = rho * np.exp(-1j * (2 * np.pi * acq.TE) * df).astype(np.complex64)

    # ---- slice selection (through-plane weighting) ----
    thick = acq.slice_fwhm or vox[sa]
    zc = coords[sa]
    zs = np.arange(n_mr[sa]) * vox[sa]
    if acq.rf_bw is not None:
        gss = acq.rf_bw / vox[sa]  # Hz/mm
        zeff = zc.reshape([-1 if a == sa else 1 for a in range(3)]) + acq.sign_ss * df / gss
    else:
        zeff = np.broadcast_to(zc.reshape([-1 if a == sa else 1 for a in range(3)]), m.shape)
    m = np.moveaxis(m, sa, -1)
    zeff = np.moveaxis(np.asarray(zeff), sa, -1)
    if acq.readout_axis is not None:
        dfm = np.moveaxis(df, sa, -1)
    S = np.empty(m.shape[:2] + (n_mr[sa],), np.complex64)
    S0 = np.empty(m.shape[:2] + (n_mr[sa],), np.float32)
    zc0 = zc  # undisturbed
    for k, z in enumerate(zs):
        w = _slice_profile(zeff - z, thick, acq.slice_profile)
        S[..., k] = np.sum(w * m, axis=-1)
        S0[..., k] = np.sum(_slice_profile(zc0 - z, thick, acq.slice_profile))
    # S: in-plane fine grid x slices (before readout deposition)

    # ---- readout displacement (deposit on fine in-plane grid) ----
    inplane_axes = [a for a in range(3) if a != sa]
    if acq.readout_axis is not None:
        ro = inplane_axes.index(acq.readout_axis)
        osr = os[acq.readout_axis]
        # displacement per fine cell in fine-grid units, evaluated per slice using the
        # slice-weighted mean frequency is not exact; instead deposit before summation.
        S = np.zeros_like(S)
        nro = m.shape[ro]
        for k, z in enumerate(zs):
            w = _slice_profile(zeff - z, thick, acq.slice_profile)
            shift = acq.sign_ro * dfm / acq.bw_px * osr  # fine cells
            src = np.arange(nro).reshape([-1 if a == ro else 1 for a in range(3)])
            dst = np.rint(src + shift).astype(np.int64)
            valid = (dst >= 0) & (dst < nro)
            contrib = np.where(valid, w * m, 0)
            dst = np.clip(dst, 0, nro - 1)
            # sum over z first per destination: use add.at on flattened indices
            other = 1 - ro
            idx_other = np.broadcast_to(np.arange(m.shape[other]).reshape(
                [-1 if a == other else 1 for a in range(3)]), m.shape)
            if ro == 0:
                flat = dst * m.shape[1] + idx_other
            else:
                flat = idx_other * m.shape[1] + dst
            acc = np.zeros(m.shape[0] * m.shape[1], np.complex128)
            np.add.at(acc, flat.ravel(), contrib.ravel())
            S[..., k] = acc.reshape(m.shape[:2])

    # ---- in-plane PSF ----
    na, nb = n_mr[inplane_axes[0]], n_mr[inplane_axes[1]]
    osa, osb = os[inplane_axes[0]], os[inplane_axes[1]]
    Sn = S / S0  # normalised so that no-source = 1 at every fine in-plane cell
    if acq.blur_fwhm > 0 or acq.blur_fwhm_z > 0:
        from scipy.ndimage import gaussian_filter
        sig = [acq.blur_fwhm / 2.3548 / h[a] for a in inplane_axes]
        sig.append(acq.blur_fwhm_z / 2.3548 / vox[sa])
        d = Sn - 1.0
        Sn = 1.0 + (gaussian_filter(d.real, sig, mode="constant")
                    + 1j * gaussian_filter(d.imag, sig, mode="constant"))
    if acq.psf == "box":
        A = Sn.reshape(na, osa, nb, osb, -1).mean(axis=(1, 3))
    elif acq.psf == "sinc":
        d = Sn - 1.0
        F = sfft.fft2(d, axes=(0, 1), workers=-1)
        fa = sfft.fftfreq(d.shape[0], d=1.0 / osa)  # cycles per MR pixel
        fb = sfft.fftfreq(d.shape[1], d=1.0 / osb)
        ka = np.abs(fa) <= 0.5 * acq.acq_frac[0]
        kb = np.abs(fb) <= 0.5 * acq.acq_frac[1]
        F *= (ka[:, None] & kb[None, :])[..., None]
        d = sfft.ifft2(F, axes=(0, 1), workers=-1)
        # sample at MR pixel centres (fine index os//2 within each MR pixel if odd os)
        ca = np.arange(na) * osa + osa // 2
        cb = np.arange(nb) * osb + osb // 2
        A = 1.0 + d[np.ix_(ca, cb)]
    else:
        raise ValueError(acq.psf)
    A = np.moveaxis(A, -1, sa)  # back to (i, j, k) ordering
    if return_fine:
        return A.astype(np.complex64), dict(occ=occ_tot, df=df, coords=coords)
    return A.astype(np.complex64)


def occupancy_on_mr(occ_fine, n_mr, acq: Acquisition, grid: Grid = Grid()):
    """Partial-volume fraction of the source in each MR voxel (box average)."""
    os = [grid.os_xy] * 3
    os[acq.slice_axis] = grid.os_z
    return occ_fine.reshape(n_mr[0], os[0], n_mr[1], os[1], n_mr[2], os[2]).mean(axis=(1, 3, 5))


# --------------------------------------------------------------------------
# Convenience constructors
# --------------------------------------------------------------------------

def gold_marker(axis, dchi=-25.0, radius=0.5, length=3.0, center=None):
    def build(c):
        f, ext = cylinder_inside(c, axis, radius, length)
        return dict(inside=f, extent=ext, dchi=dchi, rho=0.0, center=np.asarray(c, float))
    return build


def apply_attenuation(img, A, sigma, rng):
    """|A*I + n|, n ~ CN(0, sigma^2 (1-|A|^2)_+) per channel. img, A, sigma broadcastable."""
    a2 = np.abs(A) ** 2
    var = np.clip(1.0 - a2, 0.0, None) * sigma ** 2
    sd = np.sqrt(var)
    n = sd * (rng.standard_normal(A.shape) + 1j * rng.standard_normal(A.shape))
    return np.abs(A * img + n)


def b0_dir_voxel_frame(affine):
    """B0 (scanner z) expressed in the voxel-axis mm frame of the image."""
    R = np.asarray(affine)[:3, :3]
    R = R / np.linalg.norm(R, axis=0)  # columns: voxel axes in scanner (RAS) coords
    z = np.array([0.0, 0.0, 1.0])
    return R.T @ z  # components along voxel axes


def axis_from_angles(theta_deg, az_deg, b0):
    """Unit vector at polar angle theta (deg) from b0 and azimuth az (deg) about it."""
    b = np.asarray(b0, float) / np.linalg.norm(b0)
    t = np.array([1.0, 0, 0]) if abs(b[0]) < 0.9 else np.array([0, 1.0, 0])
    e1 = np.cross(b, t)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(b, e1)
    th, az = np.radians(theta_deg), np.radians(az_deg)
    return np.cos(th) * b + np.sin(th) * (np.cos(az) * e1 + np.sin(az) * e2)
