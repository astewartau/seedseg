# Synthetic demo cases

Public demo data for SeedSeg that contains no patient images: T1w FLASH-like volumes
rendered from population models, with gold markers and distractors inserted by the
physics model in `scripts/synth`. The published set is the Hugging Face dataset
[ash347/seedseg-demo](https://huggingface.co/datasets/ash347/seedseg-demo); its card is
`DATASET_CARD.md`.

| Step | Script | Output (private = patient-derived, stays on the cluster) |
|---|---|---|
| 1 | `anatomy.py` | per-subject tissue label maps on a 1 mm grid from the RTSTRUCT contours (private) |
| 2 | `atlas.py` | probabilistic atlas + deformation PCA (`atlas.npz`); per-subject fields and coefficients (private) |
| 3 | `appearance.py` | pooled per-class intensity, texture and noise statistics (`appearance.json`) |
| 4 | `render.py` | synthetic cases: image, shaded raw image, labels, parameters |
| 5 | `evaluate.py` | gates: released ensemble on both input paths (works); nearest-real anatomy Dice below the real 95th percentile (safe) and above its 5th (plausible) |
| 6 | `package.py` | selected cases as NIfTI, DICOM, labels and ground truth for release |
| – | `qc.py` | QC montage |

Environments: steps 1–4 and 6 use the `seedseg-synth` conda env plus SimpleITK and
pydicom; step 5 uses the ONNX Runtime env of `scripts/deploy` (CUDA for speed).

Results (atlas from 112 subjects; 42 rendered cases):

| | raw image + N4 (default pipeline) | N4-like image |
|---|---|---|
| markers found | 111/126 | 109/126 |
| false positives (top 3) | 11 | 12 |
| cases with 3/3 and no other candidate | 28/42 | 26/42 |

Almost every error is a simulated distractor larger than 2 mm ranked as a marker
(distractors of 2–3 mm become candidates in 50–78 % of cases, those below 1.5 mm never).

Safe gate: 40/42 anatomies are below the 0.71 threshold (95th percentile of the real
nearest-neighbour organ Dice); the two above it (0.714, 0.718) are close to the mean
anatomy. Plausibility floor (5th percentile, 0.55): 30/42 pass. Before `render.py`
re-centred the sampled anatomy on its prostate, some samples put the gland near the
edge of the field of view. On the raw path the plausible cases give 81/90 markers and
7 false positives; the others give 30/36 and 4.

The published 8 cases are among the 15 that pass both gates and have 3/3 markers with no
other candidate on both paths. The web app's default (network 1, no mirroring) also
finds 3/3 with exactly 3 candidates on all 8.
