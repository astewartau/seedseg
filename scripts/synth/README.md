# Synthetic gold fiducial markers in prostate T1w MRI

Physics-based insertion of gold fiducial markers (labelled) and calcification /
blood-product-like susceptibility sources (unlabelled hard negatives) into real
N4-corrected T1-weighted 2D FLASH magnitude images (Siemens Skyra 3T, TR/TE
600/6.66 ms, FA 80°, 0.69 × 0.69 × 2.0 mm, axial, HFS).

| File | Purpose |
|---|---|
| `physics.py` | Forward model: source rasterisation, dipole field, 2D GRE voxel signal |
| `insert.py` | Source specs → complex attenuation → insertion with the magnitude noise model |
| `generate.py` | Generator CLI (placement, distractors, labels, per-case JSON) |
| `real_voids.py`, `features.py` | Locate and describe voids (used identically for real and simulated) |
| `calibrate_noise.py` | Noise-floor calibration from signal-free regions |
| `validate.py`, `figures.py` | Real vs simulated validation and figures |
| `test_physics.py` | Analytic and convergence checks |
| `show_case.py` | QC figure for generated cases |

Environment: conda env `seedseg-synth` (Python 3.9; numpy, scipy, nibabel, matplotlib, pandas).

## Model (draft methods)

**Sources.** A gold marker is a solid cylinder (diameter 1.0 mm, length 3.0 mm) with
arbitrary centre and axis. Distractors are unions of 1–3 randomly rotated ellipsoids,
equivalent diameter 0.5–3 mm. Each source is rasterised as a fractional occupancy map
*o*(**r**) on a fine grid that oversamples every MR voxel 7 × 7 in-plane and 14 times
through-plane (0.098 × 0.098 × 0.143 mm). Each fine cell is itself supersampled 3³
times for partial volume. The susceptibility difference to tissue is Δχ(**r**) = Δχ·*o*(**r**).
For gold, Δχ = χ_Au − χ_tissue = −34 − (−9) = −25 ppm (SI, volume).

**Field.** The relative field perturbation is computed with the dipole kernel in
k-space on a grid zero-padded by 1.5× in each direction:

  ΔB(**r**)/B₀ = F⁻¹{ [1/3 − (**k**·**b̂**)²/|**k**|²] · F{Δχ} },

where **b̂** is the B₀ direction in the voxel frame, taken from the image affine (here the
slice axis). The off-resonance frequency is Δf = γ̄B₀·ΔB/B₀, with γ̄ = 42.577 MHz/T and B₀ = 3 T.

**Signal.** Every fine cell carries transverse magnetisation
m(**r**) = ρ(**r**)·exp(−i2πΔf(**r**)·TE), where ρ = 1 − *o* + *o*·ρ_src. Gold has ρ_src = 0,
so it gives no signal. The complex signal of slice *s* at in-plane position **x** is

  S_s(**x**) = Σ_z p(z + δz(**r**) − z_s) · m(**r**),

where p is the slice profile (boxcar of 2.0 mm by default; Gaussian optional) and
δz = Δf / G_ss is the off-resonance displacement of the excited slice
(G_ss = RF bandwidth / slice thickness). The sum over fine z captures through-slice
dephasing, which dominates for 2 mm slices. A readout displacement
δx = Δf / BW_px (pixels) is available as an option: each fine cell is deposited at its
displaced readout position. The in-plane point-spread function is the boxcar average
over the MR pixel by default; ideal k-space truncation ("sinc") and an extra Gaussian
are options. The complex voxel attenuation is A = S / S₀, where S₀ is the same
computation without the source. Each source is simulated in its own 25 × 25 × 9-voxel
patch (17 × 17 × 18 mm). Attenuations of different sources are multiplied, which is
exact when their perturbations do not overlap; the placement rules ensure that.

**Insertion and noise.** The images are multi-channel magnitude reconstructions.
Signal-free regions such as rectal gas sit at 3.6–5 × the tissue noise SD rather than
the Rician 1.25 ×. Their SD is ≈ 0.7 × the tissue SD (`noise_calibration.json`). Both
match a sum-of-squares noise floor (non-central χ with 2N degrees of freedom), with
N_eff ≈ 10. A Rician floor would make simulated voids far too dark: the real void floor
is ≈ 0.34 × the CTV median, while Rician noise would put it at ≈ 0.10. The synthetic
magnitude is therefore formed in the power domain:

  M′² = |A|²·M² + (1 − |A|²)₊ · σ² · Σ_{c=1..N} |n_c|²,  n_c ~ CN(0, 1),

with M the original N4-corrected magnitude and σ the per-channel noise SD. σ is
estimated per subject from the MAD of an in-plane Laplacian residual within the dilated
CTV. The noise is spatially correlated with lag-1 coefficients 0.38 (i) and 0.22 (j),
measured in the purest gas regions. This gives E[M′²] = |A|²S² + 2Nσ², the exact
expectation for the attenuated signal. The image is unchanged where A = 1, and the void
floor follows the scanner's noise-floor distribution. Output images keep the source's
int16 storage (with NIfTI scaling), so quantisation gives no cue for synthetic regions.

**Distractors.** 0–4 per case, inside the prostate. 75 % are diamagnetic, with
Δχ log-uniform in [−10, −1] ppm and ρ_src ∈ [0, 0.3] (calcification). 25 % are
paramagnetic, with Δχ log-uniform in [+1, +10] ppm and ρ_src ∈ [0.3, 1] (blood-product
foci). They keep the label of the voxel they replace (background or prostate). They are
at least 6 mm from synthetic marker centres and at least 3 mm from real-marker label voxels.

**Placement.** There are 1–3 synthetic markers per case. Their centres (continuous,
sub-voxel) lie ≥ 1.5 mm inside the prostate (labels 1 ∪ 2), ≥ 10 mm from any
real-marker label voxel and ≥ 8 mm from each other. Each marker gets a context with
probabilities 0.5 / 0.25 / 0.25. "random" is anywhere admissible. "boundary" is
1.5–4 mm from the gland edge. "bright" is within 4 mm of a voxel brighter than
1.6 × the CTV median (T1-bright blood products) within 6 voxels of the gland. When a
context has no admissible voxel, placement falls back to random. The polar angle to B₀
is drawn from the fitted distribution (below) plus U(−5°, 5°), with uniform azimuth.
Δχ is drawn from U(−35, −15) ppm.

**Labels.** The subject's label variant (real markers and prostate) is kept unchanged.
Synthetic markers are added as class 1, as 3 mm-radius spheres (in mm, so ellipsoidal
in voxels) at the marker centre. `--label-mode footprint` writes instead the
thresholded void (|A| < 0.5) connected to the marker, together with the gold occupancy.

## Validation (`synthetic/validation/`)

*Real reference set.* 361 markers from all 127 subjects, at the refined MR marker
positions (`labelfix/refined`, void_strength ≥ 0.45). *Simulated set.* Markers inserted
at marker-free prostate locations of the same subjects. Real and simulated voids are
located (smoothed minimum within a 3.5 mm sphere, then deficit-weighted centroid) and
described (per-slice radial profiles, per-slice minimum and void area at 0.6 × CTV
median, in-plane elongation) by exactly the same code. A library of 360 simulations
(polar angle 0–90° in 10° steps × Δχ ∈ {−15, −25, −35} ppm × 12 random positions) was
matched to each real void by nearest neighbours of its rotation-invariant descriptor.
That gives a per-void posterior over (θ, Δχ); the pooled posterior defines the
generation prior. A fresh set of 361 simulations drawn from that prior was then
compared with the real set.

Results (`metrics_final.json`, `fig_*_final.png`; medians, real vs simulated):

| | real | simulated | KS p |
|---|---|---|---|
| min intensity / CTV median | 0.34 [0.29–0.39] | 0.33 [0.30–0.36] | < 0.001 |
| void volume (mm³, < 0.6) | 41.6 [33–55] | 43.5 [30–57] | 0.12 |
| in-plane elongation | 1.19 [1.11–1.30] | 1.18 [1.11–1.28] | 0.76 |
| slices with void | 4 [3–4] | 3 [3–4] | 0.08 |
| void area per slice (mm², −2…+2 slices) | 0, 4.3, 9.5, 5.2, 0.5 | 0, 3.3, 9.9, 4.3, 0 | |

The RMS difference of the median per-slice radial profiles is 0.034 (in CTV-median units).

* The orientation likelihood is weak. Voids at 0.69 × 0.69 × 2 mm with this noise floor
  barely encode marker orientation. The pooled posterior is roughly flat over 0–60° and
  declines towards 90°. It favours orientations closer to B₀ than an isotropic sin θ
  distribution would, as expected for transperineal implantation along the
  cranio-caudal axis. Δχ is not identifiable over −15 to −35 ppm: the posterior is
  0.30 / 0.37 / 0.34. Hence the uniform Δχ prior around the nominal −25 ppm.
* Off-resonance slice displacement (RF bandwidth 1300 Hz, positive polarity) reproduces
  the real cranio-caudal asymmetry of the voids: larger deficit in the slice above the
  minimum. With the opposite polarity the asymmetry is reversed, and without
  displacement the median simulated void is symmetric (`metrics_base.json`). The RF bandwidth is not in the anonymised DICOMs, so it was
  chosen as plausible for this sequence rather than fitted.
* In exploratory runs (48 simulations per setting, run before the noise model was
  finalised), readout displacement at 80–150 Hz/px changed the void descriptors by less
  than their sampling variability. Off-resonance exceeds half a pixel only within ≈ 1.5–2 mm of the
  marker, where the signal is already dephased, and the mean real void shows no
  readout-direction pile-up. The readout bandwidth is unknown, so the effect is off by
  default (`--set readout_axis=0 --set bw_px=…`).
* Known mismatches: (i) the simulated void-floor distribution is narrower than the real
  one, because a single N_eff and a per-subject scalar σ are used, while the real
  floors vary between and within subjects; (ii) real voids extend slightly further in z
  (4 vs 3 slices); (iii) real voids sit on heterogeneous tissue and sometimes on other
  structures, and the simulated backgrounds are random marker-free prostate locations.

Numerical checks (`physics_checks.json`):

* The dipole field of a sphere matches the analytic solution to ≤ 0.012 ppm (1.5 Hz)
  outside 1.5 radii (max field 4.9 ppm, padding 1.5).
* Against 9 × 9 × 20 oversampling, the default 7 × 7 × 14 grid gives |A| errors with a
  99th percentile ≤ 0.06 inside the void (max 0.11). This is below the image noise.

## Usage

```bash
PY=~/miniconda3/envs/seedseg-synth/bin/python
# generate K cases per subject for a label variant (deterministic in --seed, sid, k)
$PY scripts/synth/generate.py --subjects all --variant corrected --k 2 --seed 0
# options: --label-mode footprint, --set readout_axis=0 --set bw_px=250, --param n_markers=[2,3]
# validation and figures
$PY scripts/synth/calibrate_noise.py
$PY scripts/synth/validate.py --variant final --set rf_bw=1300 --set sign_ss=1
$PY scripts/synth/figures.py --variant final
$PY scripts/synth/test_physics.py
```

Outputs go to `prostate-v2/synthetic/{variant}/{images,labels,params}/{sid}_syn{k}.*`.
The JSON holds every parameter needed to regenerate the case: positions, axes, Δχ,
distractor geometry, acquisition, grid, noise, seed and git revision.
