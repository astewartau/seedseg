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
| 5 | `evaluate.py` | gates: released ensemble on both input paths (works) and nearest-real anatomy Dice (safe) |
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
All 42 anatomies pass the safe gate: the highest mean organ Dice to any source anatomy
is 0.55–0.70 against a 0.71 threshold (95th percentile of real nearest-neighbour Dice).
