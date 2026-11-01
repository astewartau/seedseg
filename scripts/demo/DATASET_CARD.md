---
license: cc-by-4.0
pretty_name: SeedSeg synthetic demo data
tags:
- medical-imaging
- mri
- prostate
- fiducial-markers
- synthetic
size_categories:
- n<1K
---

# SeedSeg synthetic demo data

Synthetic T1-weighted prostate MRI with three gold fiducial markers each, for trying
[SeedSeg](https://github.com/astewartau/seedseg) (web app, Python pipeline or OpenRecon
container) without patient data. **These images are synthetic. They are not patient
scans.**

Each case imitates the axial 2D FLASH acquisition SeedSeg was trained on (3 T,
TR/TE 600/6.66 ms, flip angle 80°, 0.6875 × 0.6875 × 2 mm, 320 × 320 × 40). The images
contain three gold markers and one to three simulated calcification-like or
blood-product-like distractors in the prostate.

## Files

| File | Content |
|---|---|
| `demo-XXX_T1w.nii.gz` | T1w magnitude as from the scanner (with receive-coil shading; SeedSeg applies N4 itself) |
| `demo-XXX_T1w.json` | acquisition parameters |
| `demo-XXX_labels.nii.gz` | 1 = gold marker (3 mm sphere at the marker centre), 2 = prostate |
| `demo-XXX_truth.json` | marker centres (voxel and RAS mm), axis, angle to B0, susceptibility; distractor positions and type |
| `dicom/demo-XXX/` | the same image as a DICOM MR series |

## Using the data

- Web app: open https://astewartau.github.io/seedseg/ and click **Load example data**,
  or drop a `*_T1w.nii.gz` file or a `dicom/demo-XXX` folder.
- Python: `python scripts/deploy/seedseg_onnx.py --models <dir with seedseg-v2.json> --input demo-001_T1w.nii.gz --out results/demo-001`
- Compare the detections with `demo-XXX_truth.json`.

## How the images were made

- **Anatomy.** Organ contours from radiotherapy planning (body, bone, bladder, rectum,
  prostate, seminal vesicles, penile bulb, rectal spacer) of 112 prostate patients
  were combined into a probabilistic atlas with a model of shape variation. Each case
  is a new anatomy drawn from that population model.
- **Appearance.** Tissue intensities, texture and noise use population statistics from
  the same acquisitions, with a multi-coil magnitude noise model, thin dark lines at
  tissue interfaces, bright inflow vessels and receive-coil shading.
- **Markers and distractors.** Gold markers (1 × 3 mm cylinders) and distractors were
  inserted with a physics model: dipole field, through-slice dephasing, off-resonance
  slice displacement and the noise floor. The model was validated against 361 real
  marker voids.

No voxel of any patient image is in this dataset. Only label-derived population
models and summary statistics were used. Each synthetic anatomy was compared with
every source anatomy (mean Dice over bladder, rectum, prostate, bone marrow and
seminal vesicles, aligned at the prostate). No case is closer to any source patient
than the 95th percentile of how close the source patients are to each other.

## Detection performance

These 8 cases were selected from 42 generated cases. In each of them the released
SeedSeg ensemble finds all three markers and no other candidate. The web app's default
setting (one network, no test-time mirroring) gives the same result on all 8.

Over all 42 generated cases, the released 5-fold ensemble found 111 of 126 markers,
with 11 false positives among the top-3 detections. Almost every error came from a
simulated distractor larger than 2 mm being ranked as a marker. In cross-validation on
the 127 real patients, the same model configuration found 354 of 377 markers with 9 false
positives. The simulated distractors are therefore harder than typical real
calcifications.

## Licence

CC BY 4.0. Please cite SeedSeg (see the repository) when using these data.
