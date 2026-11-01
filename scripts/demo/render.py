#!/usr/bin/env python3
"""Step 4: render synthetic T1w FLASH cases from the population models, then insert
gold markers and distractors with the validated physics generator (scripts/synth).

Per case (deterministic in --seed and the case index):
  shape       z_i ~ N(0, 1) truncated to |z| <= Z_MAX for the first N_MODES deformation
              modes; phi = mean + sum z_i sqrt(lambda_i) e_i (subject grid -> template)
  anatomy     class weights w_c(y) = P_c(y + phi(y))^GAMMA, normalised, sampled on a fine
              grid (native in-plane 0.6875 mm, 4 sub-slices per 2 mm slice)
  appearance  log(mu_c, noise) ~ population multivariate normal; per class a smooth
              low-pass field (5 mm) and a G1 - G5 texture band with lognormal SDs
  signal      S_target = sum_c w_c (mu_c + low_c + tex_c); boxcar slice profile (mean of
              the sub-slices); S = sqrt(max(S_target^2 - 2 N sigma^2, 0)) so that the
              sum-of-squares magnitude below has E[M^2] = S_target^2
  air         outside the body the magnitude is scaled down to the population air level
  noise       M = | sum-of-squares over N channels of (S [channel 1] + sigma n_c) |,
              N ~ U{6..14}, lag-1 noise correlation (0.38, 0.22), as in scripts/synth
  markers     scripts/synth/generate.make_case on the rendered image with the prostate
              label (3 markers, 1-3 distractors)
  raw         the final image times a smooth receive-coil shading field (for the
              default pipeline, which applies N4 itself)

Outputs in --out: images/{case}.nii.gz (N4-like), raw/{case}.nii.gz (shaded),
labels/{case}.nii.gz (1 markers, 2 prostate), params/{case}.json.
"""
import argparse, json, os, sys, time
import numpy as np
import nibabel as nib
import scipy.ndimage as ndi

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "synth"))
import anatomy as A  # noqa: E402

SHAPE = (320, 320, 40)
VOX = np.array([0.6875, 0.6875, 2.0])
SUB = 4
# rendering knobs (render.py --set NAME=value)
KNOBS = dict(
    N_MODES=10, Z_MAX=2.0,
    GAMMA=6.0,              # class-weight sharpening: w_c ~ P_c^GAMMA (large -> hard argmax)
    TEX_FINE=0.5,           # texture band G(TEX_FINE) - G(5 mm)
    INTERFACE=(0.25, 0.45),  # darkening of in-plane tissue interfaces (fat/water cancellation lines)
    VESSELS=(2, 6),         # bright through-plane flow vessels per case
    VESSEL_R=(1.0, 3.5), VESSEL_I=(1.6, 2.8),
)
SOFT = None
REF_UNITS = 1000.0


def grf(shape, vox, rng, s_lo, s_hi=None):
    """Unit-SD Gaussian random field: white noise -> G(s_lo) [- G(s_hi)] (mm)."""
    n = rng.standard_normal(shape).astype(np.float32)
    f = ndi.gaussian_filter(n, np.array(s_lo) / vox)
    if s_hi is not None:
        f = f - ndi.gaussian_filter(n, np.array(s_hi) / vox)
    return f / max(float(f.std()), 1e-6)


def draw_appearance(app, rng):
    k = len(app["classes"])
    v = rng.multivariate_normal(app["log_mu_mean"], app["log_mu_cov"])
    mu = np.exp(v[:k]); noise = float(np.exp(v[k]))
    low = np.exp(rng.normal(app["log_low_mean"], app["log_low_sd"]))
    tex = np.exp(rng.normal(app["log_tex_mean"], app["log_tex_sd"]))
    return mu, noise, low, tex


def render(atlas, app, ctv_vox, rng):
    K = KNOBS
    nm = min(K["N_MODES"], len(atlas["lam"]))
    z = np.clip(rng.standard_normal(nm), -K["Z_MAX"], K["Z_MAX"])
    fshape = tuple(atlas["field_shape"])                    # sitk order (z, y, x, 3)
    field = atlas["mean"] + (z * np.sqrt(atlas["lam"][:nm])) @ atlas["modes"][:nm]
    field = field.reshape(fshape).transpose(2, 1, 0, 3)     # (x, y, z, 3), mm
    G = np.array(atlas["grid_mm"]); ds = float(atlas["field_spacing"])

    # the sampled deformation can move the prostate off the grid centre; shift the render
    # window so the prostate lands at the intended position in the field of view
    P = atlas["P"]
    gp = np.stack(np.meshgrid(*[np.arange(0, n, 2, dtype=np.float32) for n in G], indexing="ij"), 0)
    up = np.stack([ndi.map_coordinates(field[..., k], gp / ds, order=1, mode="nearest") for k in range(3)], 0)
    pp = ndi.map_coordinates(P[A.C["prostate"]].astype(np.float32), (gp + up).reshape(3, -1), order=1).reshape(gp.shape[1:])
    shift = (np.array(ndi.center_of_mass(pp)) * 2 - (G - 1) / 2) if pp.sum() > 0 else np.zeros(3)
    del gp, up, pp

    # fine render grid -> subject 1 mm grid index coordinates
    fine = (SHAPE[0], SHAPE[1], SHAPE[2] * SUB)
    fvox = np.array([VOX[0], VOX[1], VOX[2] / SUB])
    off = [(np.arange(fine[0]) - ctv_vox[0]) * VOX[0],
           (np.arange(fine[1]) - ctv_vox[1]) * VOX[1],
           ((np.arange(fine[2]) + 0.5) / SUB - 0.5 - ctv_vox[2]) * VOX[2]]
    Y = np.stack(np.meshgrid(*[o + (n - 1) / 2 + sh for o, n, sh in zip(off, G, shift)], indexing="ij"), 0).astype(np.float32)
    U = np.stack([ndi.map_coordinates(field[..., k], Y / ds, order=1, mode="nearest") for k in range(3)], 0)
    Tc = (Y + U).reshape(3, -1)
    del Y, U

    mu, noise, low, tex = draw_appearance(app, rng)
    W = np.zeros((P.shape[0],) + fine, np.float32)
    for c in range(P.shape[0]):
        W[c] = ndi.map_coordinates(P[c].astype(np.float32) / 255, Tc, order=1, mode="nearest").reshape(fine)
    del Tc
    prost_p = W[A.C["prostate"]] / np.maximum(W.sum(0), 1e-6)
    # sharpen structures, not the soft-tissue intensity sub-classes: the soft classes are
    # sharpened as one group, which is then shared among them by their atlas probabilities
    si = [A.C["soft_dark"], A.C["soft_mid"], A.C["soft_bright"]]
    Wsoft = W[si] / np.maximum(W[si].sum(0), 1e-6)
    Hs = np.array(si, np.uint8)[Wsoft.argmax(0)]
    W[si[0]] = W[si].sum(0); W[si[1:]] = 0
    W = W ** K["GAMMA"]
    W /= np.maximum(W.sum(0), 1e-6)
    W[si] = W[si[0]] * Wsoft
    del Wsoft
    S = np.zeros(fine, np.float32)
    for c in range(P.shape[0]):
        if W[c].max() < 1e-3:
            continue
        lowf = grf(tuple(np.array(fine) // 4), fvox * 4, rng, 5.0)
        lowf = ndi.zoom(lowf, np.array(fine) / np.array(lowf.shape), order=1)
        texf = grf(fine, fvox, rng, K["TEX_FINE"], 5.0)
        S += W[c] * (mu[c] + low[c] * lowf + tex[c] * texf)
    H = W.argmax(0).astype(np.uint8)
    H = np.where(np.isin(H, si), Hs, H)
    extra = {}
    if K["INTERFACE"]:
        # in-plane class interfaces inside the body: thin dark lines (partial fat/water cancellation)
        body = (H != A.C["air"]) & (H != A.C["gas"])
        bnd = np.zeros(fine, bool)
        for ax in (0, 1):
            d = np.diff(H, axis=ax) != 0
            sl0 = [slice(None)] * 3; sl0[ax] = slice(0, -1)
            bnd[tuple(sl0)] |= d
        bnd &= body
        kappa = rng.uniform(*K["INTERFACE"])
        S *= 1 - kappa * np.clip(ndi.gaussian_filter(bnd.astype(np.float32), (0.5, 0.5, 0)) * 2.5, 0, 1)
        extra["interface_kappa"] = float(kappa)
    if K["VESSELS"]:
        extra["vessels"] = add_vessels(S, H, fvox, rng, K)
    S = np.clip(S, 0, None).reshape(SHAPE[0], SHAPE[1], SHAPE[2], SUB).mean(-1)
    S = ndi.gaussian_filter(S, (0.4, 0.4, 0))
    prost = prost_p.reshape(SHAPE[0], SHAPE[1], SHAPE[2], SUB).mean(-1) >= 0.5
    w_air = W[A.C["air"]].reshape(SHAPE[0], SHAPE[1], SHAPE[2], SUB).mean(-1)
    # class map on the 1 mm subject grid (for the privacy comparison with real anatomy)
    Y1 = np.stack(np.meshgrid(*[np.arange(n, dtype=np.float32) for n in G], indexing="ij"), 0)
    U1 = np.stack([ndi.map_coordinates(field[..., k], Y1 / ds, order=1, mode="nearest") for k in range(3)], 0)
    T1 = (Y1 + U1).reshape(3, -1)
    lab1 = np.argmax(np.stack([ndi.map_coordinates(P[c], T1, order=1, mode="nearest") for c in range(P.shape[0])]), 0)
    lab1 = lab1.reshape(tuple(G)).astype(np.uint8)
    return S, prost, lab1, w_air, dict(shape_z=z.tolist(), mu=dict(zip(A.CLASSES, mu.tolist())), noise=noise,
                          low=dict(zip(A.CLASSES, low.tolist())), tex=dict(zip(A.CLASSES, tex.tolist())),
                          knobs={k: v for k, v in K.items()}, prostate_shift_mm=shift.tolist(), **extra)


def add_vessels(S, H, fvox, rng, K):
    """Bright through-plane vessels (inflow enhancement in 2D FLASH): straight cylinders,
    mostly along z (tilt <= 20 deg), in soft tissue at least 45 mm from the prostate."""
    soft = np.isin(H, [A.C["soft_dark"], A.C["soft_mid"], A.C["soft_bright"]])
    pr = H == A.C["prostate"]
    pc = np.array(np.nonzero(pr)).mean(1) if pr.any() else np.array(H.shape) / 2
    mid = H.shape[2] // 2
    cand = np.argwhere(soft[:, :, mid])
    cand = cand[np.hypot(*((cand - pc[:2]) * fvox[:2]).T) > 45]
    out = []
    if not len(cand):
        return out
    g = np.meshgrid(*[np.arange(n, dtype=np.float32) * v for n, v in zip(H.shape, fvox)], indexing="ij")
    for _ in range(int(rng.integers(K["VESSELS"][0], K["VESSELS"][1] + 1))):
        c = cand[rng.integers(len(cand))] * fvox[:2]
        r = rng.uniform(*K["VESSEL_R"]); inten = rng.uniform(*K["VESSEL_I"])
        tilt = np.radians(rng.uniform(0, 20)); az = rng.uniform(0, 2 * np.pi)
        dz = g[2] - mid * fvox[2]
        cx = c[0] + np.tan(tilt) * np.cos(az) * dz; cy = c[1] + np.tan(tilt) * np.sin(az) * dz
        w = np.clip((r - np.hypot(g[0] - cx, g[1] - cy)) / 0.5 + 0.5, 0, 1) * soft
        S *= 1 - w; S += w * inten
        out.append(dict(centre_mm=c.tolist(), radius_mm=r, intensity=inten, tilt_deg=float(np.degrees(tilt))))
    return out


def add_noise(S, sigma, n_coils, rng, corr=(0.38, 0.22)):
    from insert import correlated_noise
    S = np.sqrt(np.clip(S ** 2 - 2 * n_coils * sigma ** 2, 0, None))
    m2 = np.abs(S + sigma * correlated_noise(S.shape, rng, corr)) ** 2
    for _ in range(n_coils - 1):
        m2 += sigma ** 2 * np.abs(correlated_noise(S.shape, rng, corr)) ** 2
    return np.sqrt(m2).astype(np.float32)


def coil_shading(rng):
    """Smooth multiplicative receive field: anterior/posterior coil gain + random smooth part."""
    x, y, z = np.meshgrid(*[np.linspace(-1, 1, n) for n in SHAPE], indexing="ij")
    ap_ = rng.uniform(0.15, 0.35)
    f = ap_ * (y ** 2) + rng.uniform(-0.1, 0.1) * x + rng.uniform(-0.08, 0.08) * z
    g = grf(tuple(np.array(SHAPE) // 8), VOX * 8, rng, 40.0)
    f += 0.12 * ndi.zoom(g, np.array(SHAPE) / np.array(g.shape), order=1)
    return np.exp(f - f.mean()).astype(np.float32)


def zero_frame(rng):
    """The scanner's zero-valued in-plane frame: 1-3 columns at each x edge, and 1-2 rows
    at each y edge that widen towards the corners (as in the real images)."""
    m = np.zeros(SHAPE[:2], bool)
    a, b = rng.integers(1, 4, 2)
    m[:a] = True; m[SHAPE[0] - b:] = True
    x = np.linspace(-1, 1, SHAPE[0])
    for side in (0, 1):
        w = np.round(rng.uniform(1, 2) + rng.uniform(1, 4) * x ** 4).astype(int)
        for i, wi in enumerate(w):
            if side == 0:
                m[i, :wi] = True
            else:
                m[i, SHAPE[1] - wi:] = True
    return m


def affine():
    a = np.diag([-VOX[0], VOX[1], VOX[2], 1.0])
    a[:3, 3] = -a[:3, :3] @ ((np.array(SHAPE) - 1) / 2)
    return a


def make(case, seed, atlas, app, ctv_vox, out, stage):
    import generate as GEN
    t0 = time.time()
    rng = np.random.default_rng([seed, int(case.split("-")[-1])])
    S, prost, lab1, w_air, rp = render(atlas, app, ctv_vox, rng)
    os.makedirs(f"{out}/anat1mm", exist_ok=True)
    A1 = np.eye(4); A1[:3, 3] = -(np.array(lab1.shape) - 1) / 2
    nib.save(nib.Nifti1Image(lab1, A1), f"{out}/anat1mm/{case}_lab.nii.gz")
    n_coils = int(rng.integers(6, 15))
    img = add_noise(S, rp["noise"], n_coils, rng)
    # air outside the body is darker in the scanner images than the sum-of-squares floor
    f_air = min(rp["mu"]["air"] / (np.sqrt(2 * n_coils) * rp["noise"]), 1.0)
    img = img * (1 - w_air * (1 - f_air)) * REF_UNITS
    frame = zero_frame(rng)
    img[frame] = 0
    prost[frame] = False
    aff = affine()
    os.makedirs(f"{stage}/images", exist_ok=True); os.makedirs(f"{stage}/labels/demo", exist_ok=True)
    hdr = nib.Nifti1Header(); hdr.set_data_dtype(np.int16)
    nib.save(nib.Nifti1Image(img, aff, hdr), f"{stage}/images/{case}.nii")
    nib.save(nib.Nifti1Image((prost * 2).astype(np.uint8), aff), f"{stage}/labels/demo/{case}.nii.gz")

    GEN.ROOT = stage
    if not getattr(GEN, "_demo_rng", False):     # the generator's RNG expects numeric subject ids
        orig = GEN.case_rng
        GEN.case_rng = lambda seed_, sid, k: orig(seed_, str(int(sid.split("-")[-1]) + 10 ** 6), k)
        GEN._demo_rng = True
    P = dict(GEN.DEFAULTS, n_markers=(3, 3), n_calcs=(1, 3), n_coils=(n_coils, n_coils))
    GEN.make_case(case, 0, "demo", seed, out, "sphere", dict(GEN.DEFAULT_ACQ), P)
    for sub in ("images", "labels", "params"):
        ext = "json" if sub == "params" else "nii.gz"
        os.replace(f"{out}/{sub}/{case}_syn0.{ext}", f"{out}/{sub}/{case}.{ext}")
    final = nib.load(f"{out}/images/{case}.nii.gz")
    raw = np.asarray(final.dataobj, np.float32) * coil_shading(rng)
    raw[frame] = 0
    os.makedirs(f"{out}/raw", exist_ok=True)
    nib.save(nib.Nifti1Image(raw, aff, hdr), f"{out}/raw/{case}.nii.gz")
    rec = json.load(open(f"{out}/params/{case}.json"))
    rec.update(render=rp, n_coils_render=n_coils, ctv_centre_vox=list(map(float, ctv_vox)),
               render_runtime_s=time.time() - t0)
    json.dump(rec, open(f"{out}/params/{case}.json", "w"), indent=1)
    return case, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--atlas", required=True, help="dir with atlas.npz")
    ap.add_argument("--appearance", required=True)
    ap.add_argument("--anat", required=True, help="anatomy.py output (for the population CTV position)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--nproc", type=int, default=4)
    ap.add_argument("--set", action="append", default=[], help="rendering knob NAME=json")
    a = ap.parse_args()
    for kv in a.set:
        k, v = kv.split("=", 1)
        if k not in KNOBS:
            raise SystemExit(f"unknown knob {k}")
        KNOBS[k] = json.loads(v)
    atlas = dict(np.load(f"{a.atlas}/atlas.npz"))
    app = json.load(open(a.appearance))
    import glob
    cv = np.median([json.load(open(f))["ctv_centre_vox"] for f in glob.glob(f"{a.anat}/*.json")], 0)
    stage = f"{a.out}/_stage"
    cases = [f"demo-{i:03d}" for i in range(a.start, a.start + a.n)]
    from multiprocessing import Pool
    with Pool(a.nproc) as p:
        for r in p.starmap(make, [(c, a.seed, atlas, app, cv, a.out, stage) for c in cases]):
            print(*r, flush=True)


if __name__ == "__main__":
    main()
