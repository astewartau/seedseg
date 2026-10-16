# SeedSeg v2: plan and status

The goal is to improve gold-marker detection in prostate T1 MRI beyond the
preprint (bioRxiv 10.64898/2026.06.18.733061). The approach has four parts:
fix the reference labels, retrain with nnU-Net, add physics-simulated markers,
and evaluate with cross-validation over all 127 subjects.

Last updated: 2026-10-16 (Claude session on bun154).

## Why (diagnosis, 2026-10-16)

The released 4-model ensemble reproduces the preprint (test 70/73 sensitivity,
70/74 precision). On top of that:
- **The models underfit.** They detect only 93% of markers on their own clean
  training subjects. The best single model (seed 123) equals the ensemble on test.
- **Checkpoint choice.** Early stopping watched the loss on 9 validation subjects,
  so the kept checkpoints come from epochs 163–368, while the learning rate was
  still 82–96% of its maximum.
- **Bad labels.** About 14% of the training labels were wrong: 13 subjects were
  12–55 mm off, 9 markers were missing across 7 subjects, and the CTV was wrong
  in those same subjects. The cause was a faulty earlier derivation, not the
  clinical registration.
- **Remaining errors.** These are confident look-alikes (calcification, blood)
  outranking true markers, plus 2 markers next to T1-bright blood that no model
  detects.
- **Test set too small.** At n=25, with only ~5 errors left, it can't show
  further improvement, hence the cross-validation over all 127 subjects.

## Workstreams and status

| # | Workstream | Status | Where |
|---|---|---|---|
| 1 | Rebuild labels from DICOM, refine on MR voids, QC | **Done**; methods draft written | `scripts/labels/` (README = methods), branch `v2-labels` |
| 2 | Centroid-matching evaluation (5 mm, one-to-one, CIs) | **Done** | `scripts/evaluation/evaluate_detections.py` |
| 3 | nnU-Net v2 ResEnc-M, 5-fold CV | **Queued on GPU** (nothing has trained yet) | `scripts/nnunet/` in worktree `seedseg-nnunet`, branch `v2-nnunet` |
| 4 | Physics-based synthetic markers and distractors | **Done and validated**; 254 cases generated | `scripts/synth/` (README = methods), worktree `seedseg-synth`, branch `v2-synth` |
| 5 | Second-stage candidate re-ranker | **Code done**; no gain on the old models' candidates (not a fair test) | `scripts/rerank/rerank.py` |
| 6 | Write-up (methods/results) | Not started | – |

## Experiment arms (nnU-Net)

Dataset ID = 500 + 10·variant + design code. Variants: original 0, corrected 1,
original_sphere 2. Design codes: cvall 1, temporal 2, cvall+syn 3, temporal+syn 4.
Fold assignment is fixed in `scripts/nnunet/splits/` and shared by all variants.

| Dataset | Labels | Design | Purpose | Slurm job (2026-10-16) |
|---|---|---|---|---|
| 511 | corrected | 5-fold CV, all 127 | **main result** | 29915715 |
| 501 | original | 5-fold CV, all 127 | label-fix ablation | 29915716 |
| 512 | corrected | temporal (102 train/val, 25 test) | comparison with the preprint split | 29915717 |
| 521 | original_sphere | 5-fold CV | shape vs position control | 29915718 |
| 513 | corrected + 254 synthetic | 5-fold CV | synthetic-data arm | preprocess 29925248, then train 29925249 |
| – | 501, fold 0, 5 epochs | smoke test | per-epoch time, export check | 29913696 |

- All jobs run on H100 (`a_ai_collab`, `gpu_cuda`), with a 72 h limit per fold.
- Only 4 GPU jobs run at once per user.
- The estimate is about 1–2 days per fold. The smoke test gives the real figure.
- If a job dies, resubmitting it resumes from `checkpoint_latest`; see `train.slurm`.

## Baseline numbers for the new evaluation (released models vs corrected labels, 5 mm matching)

| Set | Sensitivity | Precision | Perfect |
|---|---|---|---|
| Test (25) | 70/74 | 70/75 | 20/25 |
| Val (9) | 25/27 | 25/26 | 7/9 |
| All 127 (partly in-sample for the old models) | 342/378 (90.5%) | 342/377 | 93/127 |

## Locations

- Data root: `/scratch/user/uqaste15/prostate-v2/`
  - `images/{sid}.nii`: N4-corrected T1 (symlinks into `raw/`).
  - `labels/{original,corrected,original_sphere}/{sid}.nii.gz`: labels (0 background, 1 marker, 2 prostate).
  - `labelfix/`: derived positions (`markers/`), refinement records (`refined/`), QC pages (`qc/`), CSVs.
  - `dicoms/`: extracted per-subject archives (MR/CT/RS/RE).
  - `synthetic/corrected/{images,labels,params}/`, plus `synthetic/validation/` (figures and metrics).
  - `nnunet/{nnUNet_raw,nnUNet_preprocessed,nnUNet_results}/`; `logs/`; `eval/`; `cache/voidmap/`.
- Sources:
  - Data backup: `/QRISdata/Q0748/data/prostate/2026-02-19-prostate-project.tar.gz`.
  - DICOM archives: `/QRISdata/Q0748/data/prostate/dicoms-organized/`.
  - Released ONNX models: OSF f7hjv.
- Repo worktrees (none pushed yet), all from github astewartau/seedseg:
  - `/scratch/user/uqaste15/src/seedseg` → `v2-labels` (labels, evaluation, re-ranker, this file)
  - `/scratch/user/uqaste15/src/seedseg-nnunet` → `v2-nnunet`
  - `/scratch/user/uqaste15/src/seedseg-synth` → `v2-synth`
- Conda envs:
  - `seedseg-nnunet`: nnunetv2 2.8.1, torch 2.6 cu124.
  - `seedseg-synth`.
- The label, evaluation and re-ranker scripts were run with a Python that has
  pydicom, nibabel, scipy and sklearn. The `qsmxt` env has all of these.

## How to resume

```bash
squeue -u uqaste15                       # what's still queued or running
sacct -u uqaste15 -S 2026-10-16 --format=JobID,JobName%20,State,Elapsed | grep nnunet
ls /scratch/user/uqaste15/prostate-v2/nnunet/nnUNet_results/*/*/fold_*/checkpoint_final.pth
```

When all 5 folds of a dataset have `checkpoint_final.pth`:
1. `cd /scratch/user/uqaste15/src/seedseg-nnunet && sbatch --export=ALL,DATASET=<id> scripts/nnunet/predict.slurm`.
   This writes per-subject probability maps from the held-out folds; see `scripts/nnunet/README.md`.
2. Evaluate against the corrected references (for every arm):
   `python scripts/evaluation/evaluate_detections.py --probs <marker_prob dir> --root /scratch/user/uqaste15/prostate-v2 --subjects /scratch/user/uqaste15/prostate-v2/subjects.txt --out <prefix>`
   The `--probs` directory must contain `{sid}.nii.gz` marker-class probabilities.
   `export_probs.py` writes `{sid}_marker_prob.nii.gz`, so symlink or rename those first.
3. Re-ranker on the out-of-fold candidates:
   `scripts/rerank/rerank.py --candidates <prefix>_candidates.csv --folds ../seedseg-nnunet/scripts/nnunet/splits/splits_final_Dataset511_SeedSegCvAll_corrected.json --root ... --out <x>.csv`
   (nnU-Net format: a list of 5 `{train, val}` entries), then `evaluate_detections.py --rescore <x>.csv --score rerank_p`.
4. Compare the arms. Main contrasts:
   - 511 vs 501: effect of the label fix;
   - 511 vs 521: effect of label position rather than label shape;
   - 513 vs 511: effect of synthetic data;
   - 511 single-fold vs 5-fold ensemble: whether one model is enough.

   Report sensitivity, precision, perfect subjects and confidence intervals from
   the evaluator, and also on the 25 temporal test subjects.

## Open decisions / to do

- **Human QC of the labels.** Ashley needs to review the 66 flagged markers in
  `labelfix/qc/flagged_*.png` (so far only Claude has); record the outcome in
  `labelfix/qc_decisions.csv`.
- **Backup or push.** Nothing is pushed. The label records contain patient study
  IDs, and the repo is public. Back up `prostate-v2/` and the worktrees to RDM
  (Q0748) before scratch purges them, or push the code without `scripts/labels/records/`.
- **Smoke test.** When 29913696 runs, check the epoch time and the orientation
  check in `export_probs.py`, and update the ETAs above.
- **Synthetic slice extent.** Real voids span about 4 slices, simulated about 3.
  The likely cause is the idealised boxcar slice profile. Only revisit this if
  513 beats 511.
- **Re-ranker.** If the nnU-Net models still make look-alike swaps, consider a
  patch-CNN re-ranker; the hand-crafted features added little.
- **Deployment.** If one nnU-Net model is enough, update the web app and
  OpenRecon (ONNX export plus sliding-window or whole-volume inference).
- **Paper.** The new methods are labels (`scripts/labels/README.md`), synthetic
  data (`scripts/synth/README.md`) and nnU-Net (`scripts/nnunet/README.md`).
