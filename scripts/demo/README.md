# Synthetic demo cases

Public demo data for SeedSeg that contains no patient images. The volumes imitate the
T1w 2D FLASH acquisition, are rendered from population models of anatomy and appearance,
and carry gold markers and distractors inserted by the physics model in `scripts/synth`.
The published set is the Hugging Face dataset
[ash347/seedseg-demo](https://huggingface.co/datasets/ash347/seedseg-demo) (CC BY 4.0).
Its card is `DATASET_CARD.md`. The web app's **Load example data** button loads
`demo-001_T1w.nii.gz` from it.

| Step | Script | Output (private = patient-derived, stays on the cluster) |
|---|---|---|
| 1 | `anatomy.py` | per-subject tissue label maps on a 1 mm grid from the RTSTRUCT contours (private) |
| 2 | `atlas.py` | probabilistic atlas and deformation PCA (`atlas.npz`); per-subject fields and coefficients (private) |
| 3 | `appearance.py` | pooled per-class intensity, texture and noise statistics (`appearance.json`); per-subject rows (private) |
| 4 | `render.py` | synthetic cases: N4-like image, shaded raw image, labels, parameters, 1 mm class map |
| 5 | `evaluate.py` | gates: released model on both input paths (works); nearest real anatomy (safe, plausible) |
| 6 | `package.py` | selected cases as NIfTI, DICOM, labels and ground truth |
| – | `qc.py` | QC montage |

Only steps 4–6 produce anything that is released, and only rendered cases leave the
cluster. `atlas.npz` and `appearance.json` hold population-level quantities. They are not
published.

## Methods

### 1. Anatomy (`anatomy.py`)

Subjects: the 118 with a clinical registration and an RTSTRUCT CTV in the v2 label
records (`labelfix/markers`, `labelfix/refined`).

**Grid.** A 240 × 240 × 84 mm grid at 1 mm, in the MR voxel-aligned frame, centred on the
CTV centroid of the v2 label.

**Mapping to the contours.** Each grid point is mapped to the CT frame of the RTSTRUCT in
two steps:
1. the inverse of the accepted rigid refinement of the v2 labels, if there is one;
2. the clinical spatial registration.

**Contour test.** The CT-frame point is tested against the planar contours on the nearest
contour plane, with the even-odd rule. A point more than half a plane spacing beyond an
ROI's z range is outside it.

**ROIs.** Names are matched case-insensitively (patterns in `ROI_PATTERNS`):
- body;
- bone, together with the left and right femoral heads;
- bladder;
- rectum;
- seminal vesicles;
- penile bulb;
- rectal spacer;
- gas override;
- the CTV named in the label record (the prostate).

**Excluded points.** Points outside the MR field of view get label 255. So do points on the
scanner's zero-valued in-plane frame (the image ≤ 10⁻³ of its maximum, dilated by one
voxel).

**Classes without a contour.** These are split by intensity relative to the CTV median,
r = I / median(I in CTV), using the N4-corrected T1 sampled at the same points:

| Region | Classes |
|---|---|
| inside body, outside the organ ROIs | gas (r < 0.45), else soft_dark / soft_mid / soft_bright by per-subject 1D k-means (k = 3) |
| bone ∪ femoral heads | cortex / marrow by per-subject 1D k-means (k = 2) |
| rectum | gas (r < 0.45), else rectum |
| gas override ROI | gas (except inside the prostate) |

Classes: air, soft_dark, soft_mid, soft_bright, marrow, cortex, bladder, rectum, gas,
prostate, seminal, bulb, spacer.

**Subjects kept.** Of the 118, 115 were extracted (3 failed). Three more were excluded,
leaving 112:
- one with more than 280 000 gas-class voxels outside the rectum (a large metal-implant
  void);
- two without a bone contour.

### 2. Population anatomy model (`atlas.py`)

**Registration images.** Each label map becomes a "cartoon": every class is set to its
population median intensity, the gaps outside the field of view are filled with the
nearest label, the image is smoothed with σ = 1 mm and resampled to 2 mm.

**Pairwise registration.** Template → subject, in two stages:
- *Affine:* mean squares, regular-step gradient descent, shrink factors 4/2/1, smoothing
  2/1/0 mm.
- *Fast symmetric forces demons:* 80 iterations at 4 mm, then 60 at 2 mm. Field σ is
  1.5 voxels and update-field σ is 1.0 voxel.

The affine and demons stages are composed into one dense displacement field.

**Groupwise template.**
- Start from the mean cartoon.
- Register every subject to the template.
- The new template is the field-of-view-weighted mean of the subjects' class one-hots
  warped into template space.
- Unbias the new template by resampling it with the inverse of the mean displacement.
- Run 3 rounds, then a final registration.

**Probabilistic atlas.** P_c(x) on the 1 mm grid is the field-of-view-weighted mean of the
warped one-hots. Voxels seen by fewer than 25 % of subjects take the nearest
well-covered value.

**Deformation model.**
- Each subject's field is inverted (subject grid → template), giving φ_s.
- PCA over φ_s gives a mean field, 30 modes and their eigenvalues λ_i.
- The first 10 modes explain 88 % of the variance (cumulative: 0.42, 0.52, 0.60, 0.69,
  0.75, 0.80, 0.83, 0.85, 0.87, 0.88).
- A new anatomy is P(y + φ(y)), with φ = mean + Σ zᵢ √λᵢ eᵢ.

### 3. Appearance (`appearance.py`)

**Per subject and class.** Computed inside the class mask eroded by one voxel, in r units
on the 1 mm grid:
- **μ:** median r.
- **low:** SD of the 5 mm normalised-convolution low-pass, G₅(r m) / G₅(m).
- **tex:** SD of the G₁ − G₅ band, with the image-noise contribution removed in quadrature.
  The noise leakage factor comes from passing white noise through the same native → 1 mm
  sampling and band.
- **Noise σ:** per subject, from `scripts/synth/real_voids.local_noise_sigma` in the
  dilated CTV of the native image, in r units.

**Pooled across subjects.**
- Mean and covariance of log(μ_c) and log(σ); missing classes are imputed with the
  population median.
- Lognormal fits of low and tex per class.

Population medians [IQR] of μ (CTV median = 1):

| Class | μ | Class | μ |
|---|---|---|---|
| prostate | 1.01 [1.01–1.02] | bladder | 0.77 [0.73–0.81] |
| soft_bright | 1.13 [1.11–1.20] | spacer | 0.77 [0.75–0.81] |
| bulb | 1.03 [1.01–1.10] | soft_dark | 0.62 [0.60–0.63] |
| seminal | 0.97 [0.92–0.99] | cortex | 0.38 [0.35–0.42] |
| rectum | 0.96 [0.94–0.98] | gas | 0.31 [0.26–0.34] |
| soft_mid | 0.90 [0.89–0.95] | air | 0.12 [0.09–0.17] |
| marrow | 0.84 [0.79–0.88] | noise σ | 0.08 [0.08–0.09] |

### 4. Rendering (`render.py`)

Each case is deterministic in `--seed` and its index. Defaults are in `KNOBS` and can be
changed with `--set NAME=json`.

**Output grid.** 320 × 320 × 40 voxels of 0.6875 × 0.6875 × 2 mm, axial, with the same
orientation as the scanner images. The prostate is placed at the population median CTV
position.

1. **Shape.** zᵢ ~ N(0, 1), truncated to |zᵢ| ≤ 2, for the first 10 modes. The render
   window is shifted so the prostate of the sampled anatomy lands at the target position.
   (This re-centring was added after the published set was rendered.)
2. **Class weights.** The atlas is sampled on a fine grid: native in-plane, 4 sub-slices
   per 2 mm slice. The structure classes are sharpened with w_c ∝ P_c⁶. The three soft
   classes are sharpened as one group and then share its weight by their atlas
   probabilities, so they mix smoothly instead of forming flat blocks.
3. **Appearance.** Drawn from the population models:
   - log(μ, σ) from the multivariate normal;
   - low and tex from their lognormals.

   Per class, a smooth field (5 mm) and a G₀.₅ − G₅ texture field are added, so
   S_target = Σ_c w_c (μ_c + low_c L_c + tex_c T_c).
4. **Fine structure.**
   - *Interfaces:* in-plane tissue interfaces are darkened by κ ~ U(0.25, 0.45). These are
     thin lines like the fat/water cancellation along fascia in the scanner images.
   - *Vessels:* 2–6 bright through-plane vessels: radius 1–3.5 mm, intensity 1.6–2.8, tilt
     ≤ 20°. They sit in soft tissue at least 45 mm in-plane from the prostate centre.
5. **Acquisition.**
   - *Slice profile:* boxcar, the mean of the 4 sub-slices.
   - *In-plane blur:* Gaussian, σ = 0.4 pixel.
6. **Noise.** Sum-of-squares magnitude over N ~ U{6, …, 14} channels, with lag-1 noise
   correlation (0.38, 0.22) as in `scripts/synth`.
   - Signal: S = √max(S_target² − 2Nσ², 0), so that E[M²] = S_target².
   - Air outside the body is scaled down to the population air level.
   - The zero-valued frame is reproduced: 1–3 columns at each x edge, and 1–2 rows at each
     y edge, widening towards the corners.
7. **Markers and distractors.** `scripts/synth/generate.make_case` on the rendered image
   with its prostate label: 3 markers, 1–3 distractors, other parameters as in
   `generate.DEFAULTS`. Labels: 1 = marker (3 mm sphere), 2 = prostate.
8. **Raw image.** The final image times a receive-coil shading field. SeedSeg applies N4
   itself, so this is what a user loads. The field is the exponential of:
   - an anterior–posterior quadratic, amplitude U(0.15, 0.35);
   - linear terms in x (± 0.1) and z (± 0.08);
   - a 40 mm smooth random field (SD 0.12).

### 5. Gates (`evaluate.py`)

**Works.** The released ensemble (`scripts/deploy/seedseg_onnx.py`, 5 folds, mirroring)
runs on two inputs:
- the raw image with N4 (the default pipeline);
- the N4-like image with `--no-n4`.

A marker counts as found when a top-3 detection lies within 4 mm of its centre. Any other
top-3 detection is a false positive. Extra candidate components (rank > 3) are recorded.
Per-case detections, misses and the nearest distractor or vessel to each error are written
to `detections/`.

**Safe and plausible.**
- The case's 1 mm class map is compared with every real subject's anatomy. Both are on
  prostate-centred grids and there is no registration, which favours finding similarity.
- Similarity is the mean Dice over bladder, rectum, prostate, marrow and seminal vesicles.
- **Safe:** the highest similarity to any real subject is below the 95th percentile of the
  real nearest-neighbour similarity (0.71).
- **Plausible:** it is at or above that distribution's 5th percentile (0.55).

### 6. Packaging (`package.py`)

**Selection.** Cases qualify if they:
- pass safe and plausible (plausibility was added after the published set was selected);
- have 3/3 markers with no false positive and no extra candidate on both input paths.

They are taken round-robin over the number of distractors and renumbered demo-001,
demo-002, …

**Files per case:**
- the raw image as int16 NIfTI;
- an acquisition sidecar;
- labels;
- ground-truth JSON: marker centres in voxel and RAS mm, axis, angle to B0, Δχ; distractor
  positions, type and Δχ;
- a DICOM MR series, one file per slice.

**DICOM tags.** The DICOMs carry the protocol tags the SeedSeg dicompare schema checks.
`Manufacturer` stays "SeedSeg synthetic", and ImageType is DERIVED\SECONDARY. The series
round-trips exactly (world coordinates) through dcm2niix.

## Reproducing

Paths are on Bunya. `ROOT` (`/scratch/user/uqaste15/prostate-v2`) holds the v2 images,
labels and label records. The DICOM archives are in
`/QRISdata/Q0748/data/prostate/dicoms-organized`. The work directory used was
`/scratch/user/uqaste15/demo-synth`.

### Environments

- **CPU steps (1–4, 6):** a venv on the `seedseg-synth` conda env plus SimpleITK,
  onnxruntime, pydicom, scikit-learn and huggingface_hub.
- **GPU gate (step 5):** the `scripts/deploy` ONNX Runtime GPU env, with `LD_LIBRARY_PATH`
  pointing at the NVIDIA libraries of the `seedseg-nnunet` env (`.../site-packages/nvidia/*/lib`).
  The batch size is 2: all 8 mirrorings of a full volume at once exceed H100 memory.
- **Models:** the fp32 ONNX folds from `ash347/seedseg` (`v2/fp32`, with `seedseg-v2.json`
  copied alongside).

### Commands

```bash
P=<venv>/bin/python; D=<work dir>; cd scripts/demo
$P anatomy.py    --subjects $D/subjects_all.txt   --out $D/anat  --nproc 32
#  subjects_model.txt = extracted subjects with body, bone, bladder and rectum contours
#  and no more than 280 000 gas-class voxels outside the rectum
$P appearance.py --anat $D/anat --subjects $D/subjects_model.txt --out $D/model --nproc 32
$P atlas.py      --anat $D/anat --subjects $D/subjects_model.txt --out $D/model --nproc 32
$P render.py     --atlas $D/model --appearance $D/model/appearance.json --anat $D/anat \
                 --out $D/pilot --n 42 --seed 0 --nproc 8
$P evaluate.py   --out $D/pilot --models x --anat $D/anat --subjects $D/subjects_model.txt --skip-works
$GPU_PY evaluate.py --out $D/pilot --models <onnx dir> --anat $D/anat \
                 --subjects $D/subjects_model.txt --skip-safe --device cuda --threads 8
$P package.py    --src $D/pilot --dst $D/release --n 8
```

### Runtime and resources

| Step | Resources | Time |
|---|---|---|
| anatomy | 32 cores | about 5 min (mostly reading the DICOM archives) |
| atlas | 32 cores, peak about 50 GB | about 15 min |
| render | – | about 1 min per case |
| works gate | one H100 | about 2 min per case and input path |
| safe gate | CPU | about 15 min for 42 cases |

`git_rev` in the case JSON is empty when the scripts run from a copy without `.git`.

## Published set

The published cases were rendered with seed 0 by `render.py` as of commit 2d1365b, which is
before the prostate re-centring (7b927f0). Re-rendering with the current script therefore
gives different images. The atlas and appearance models are from the 112 subjects above.

| Published | Rendered case | Distractors |
|---|---|---|
| demo-001 | demo-001 | 2 |
| demo-002 | demo-005 | 1 |
| demo-003 | demo-006 | 1 |
| demo-004 | demo-007 | 2 |
| demo-005 | demo-008 | 1 |
| demo-006 | demo-010 | 3 |
| demo-007 | demo-012 | 2 |
| demo-008 | demo-019 | 1 |

Dataset commits:
- d48dd5d: the 8 cases (the web app pins this commit);
- e60230e: DICOM series with the protocol tags; images unchanged.

Gate results over all 42 rendered cases:

| | raw image + N4 (default pipeline) | N4-like image |
|---|---|---|
| markers found | 111/126 | 109/126 |
| false positives (top 3) | 11 | 12 |
| cases with 3/3 and no other candidate | 28/42 | 26/42 |

**Errors.** Almost every error is a simulated distractor larger than 2 mm ranked as a
marker. Distractors of 2–3 mm become candidates in 50–78 % of cases; those below 1.5 mm
never do.

**Safe and plausible.**
- 40/42 anatomies are safe. The two above the threshold (0.714, 0.718) are close to the
  mean anatomy.
- 30/42 are plausible. On the raw path, the plausible cases give 81/90 markers with 7 false
  positives; the others give 30/36 with 4.
- 16 cases give 3/3 markers with no other candidate on both paths. 15 of those are also
  safe, and 11 are also plausible.
- The published 8 were selected from the 15, before the plausibility floor existed. All 8
  are plausible. They also give 3/3 markers with exactly 3 candidates under the web app's
  default (network 1, no mirroring).
