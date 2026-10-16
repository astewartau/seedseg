# SeedSeg v2: plan and status

The goal is to improve gold-marker detection in prostate T1 MRI beyond the
preprint (bioRxiv 10.64898/2026.06.18.733061). The approach has four parts:
fix the reference labels, retrain with nnU-Net, add physics-simulated markers,
and evaluate with cross-validation over all 127 subjects.

Last updated: 2026-10-17 07:05 (scheduled mid-run check, job 29940932 on bun150).

## Needs Ashley

- *(Not urgent, FYI.)* In every fold so far, the marker pseudo-Dice and the
  validation loss are best at about epoch 120–250. After that they drift down
  slowly while the train loss keeps falling: mild overfitting. For example,
  marker Dice in 511 fold 0 is 0.77 at epochs 120–130 and 0.73 at 472–482; in 501
  it is about 0.56 at epoch 250 and about 0.55 at 1000. `predict.slurm` uses
  `checkpoint_final` (the nnU-Net default). That is the unbiased choice, because
  `checkpoint_best` is picked on the held-out fold itself. I've changed nothing.
  Decide after the main result whether a shorter schedule (e.g. 250–500 epochs)
  is worth a follow-up arm.

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
| 3 | nnU-Net v2 ResEnc-M, 5-fold CV | **Training**: 501 folds 0–3 done, fold 4 running; 511 folds 0–2 running (status 2026-10-17 07:00) | `scripts/nnunet/` in worktree `seedseg-nnunet`, branch `v2-nnunet` |
| 4 | Physics-based synthetic markers and distractors | **Done and validated**; 254 cases generated | `scripts/synth/` (README = methods), worktree `seedseg-synth`, branch `v2-synth` |
| 5 | Second-stage candidate re-ranker | **Code done**; no gain on the old models' candidates (not a fair test) | `scripts/rerank/rerank.py` |
| 6 | Write-up (methods/results) | Not started | – |

## Experiment arms (nnU-Net)

Dataset ID = 500 + 10·variant + design code. Variants: original 0, corrected 1,
original_sphere 2. Design codes: cvall 1, temporal 2, cvall+syn 3, temporal+syn 4.
Fold assignment is fixed in `scripts/nnunet/splits/` and shared by all variants.

| Dataset | Labels | Design | Purpose | Slurm job (2026-10-16) |
|---|---|---|---|---|
| 511 | corrected | 5-fold CV, all 127 | **main result** | preprocess 29929992, then train 29929993 |
| 501 | original | 5-fold CV, all 127 | label-fix ablation | 29915716 |
| 512 | corrected | temporal (102 train/val, 25 test) | comparison with the preprint split | preprocess 29929994, then train 29929995 |
| 521 | original_sphere | 5-fold CV | shape vs position control | 29915718 |
| 513 | corrected + 254 synthetic | 5-fold CV | synthetic-data arm | preprocess 29929996, then train 29929997 |
| – | 501, fold 0, 5 epochs | smoke test | per-epoch time, export check | 29913696 |

- All jobs run on H100 (`a_ai_collab`, `gpu_cuda`), with a 72 h limit per fold.
- Only 4 GPU jobs run at once per user.
- **Measured: about 27–29 s per epoch, so about 8 h per 1000-epoch fold.** The smoke test
  (29913696) gave 26.9 s; the 501 folds took 7 h 53 m to 8 h 10 m.
- **ETAs (as of 2026-10-17 07:00, with 4 GPU slots and about 8 h per fold):**
  - 501: fold 4 finishes about 10:40 Sat 17 Oct; its export follows.
  - 511: folds 0–2 finish about 11:00–11:20 Sat; folds 3–4 about 19:00 Sat; export after that.
  - 512: about 11:00 Sat to about 03:00 Sun.
  - 513: about 19:00 Sat to about 11:00 Sun.
  - 521 (niced, runs last): about 11:00 Sun to about 03:00 Mon 19 Oct.

  These assume the queue starts each job promptly; partition waits add to them.
- If a job dies, resubmitting it resumes from `checkpoint_latest`; see `train.slurm`.

The corrected datasets (511/512/513) were rebuilt and re-queued on 2026-10-16 at
about 18:30, after human QC; the earlier jobs 29915715/29915717/29925249 were
cancelled. 501 started training at about 18:25.

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
1. Probability export is **already queued** and starts automatically when each
   training array finishes:
   - 501 → 29934246, 511 → 29934247, 513 → 29934248, 521 → 29934249 (CPU);
   - 512 → 29934250 (GPU, because it also predicts the 25 test subjects).

   If one is cancelled (it won't start if any fold failed), re-run it by hand:
   `cd /scratch/user/uqaste15/src/seedseg-nnunet && sbatch --export=ALL,DATASET=<id> scripts/nnunet/predict.slurm`.
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

- **Human QC of the labels: done (2026-10-16, AS).** Of the 66 flagged markers,
  only z3278008 marker 1 was rejected: no MR void at the CT-confirmed seed
  position. It carries the nnU-Net ignore label (3) and is excluded from
  evaluation (`scripts/labels/apply_qc_exclusions.py`, `records/qc_exclusions.csv`).
  The baseline on all 127 is now 342/377. Unflagged markers didn't need a
  human check. Note for the methods: the constellation-shape check only
  validates markers that were snapped to a void; the 17 low-void markers were
  covered by the visual QC instead.
- **Backup.** `scripts/backup_v2_to_rdm.sh` writes dated snapshots to
  `/QRISdata/Q0748/data/prostate/v2-work/`: a data tar, a git bundle of all
  branches, and this plan. The first snapshot was taken on 2026-10-16. **Re-run it
  after training finishes**; it picks up the trained models (checkpoint_final and
  plans). Nothing is pushed to GitHub, because the records hold patient study IDs
  and the repo is public.
- **Smoke test: done.** 29913696 completed on 16 Oct (6 min, 26.9 s per epoch; the
  first epoch takes 140 s because of compile/warm-up). It was a training-only run,
  so the orientation check in `export_probs.py` hasn't run yet. It first runs
  with the 501 export (29934246); the main_end check should look at it.
- **Synthetic slice extent.** Real voids span about 4 slices, simulated about 3.
  The likely cause is the idealised boxcar slice profile. Only revisit this if
  513 beats 511.
- **Re-ranker.** If the nnU-Net models still make look-alike swaps, consider a
  patch-CNN re-ranker; the hand-crafted features added little.
- **Deployment.** If one nnU-Net model is enough, update the web app and
  OpenRecon (ONNX export plus sliding-window or whole-volume inference).
- **Paper.** The new methods are labels (`scripts/labels/README.md`), synthetic
  data (`scripts/synth/README.md`) and nnU-Net (`scripts/nnunet/README.md`).

## Scheduled Claude checks (submitted 2026-10-16 22:20)
Launcher `scripts/monitor/claude_check.slurm` (tmux + Remote Control, opus-5-5 / medium effort, auto mode); prompts in `scripts/monitor/prompts/`. Each session appears in claude.ai/code as `seedseg-<check>-<date>` and push-notifies when done.
| Check | Job | Trigger |
|---|---|---|
| midrun | 29940932 | 4 h after 511 fold 0 starts (`after:29929993_0+240`), 6 h wall |
| main_end | 29940933 | `afterany` 511 train (29929993) + 501 train (29915716) |
| final_end | 29940934 | `afterany` 512 (29929995), 513 (29929997), 521 (29915718) |
Launcher tested 2026-10-16 (job 29940686): starts, RC registers, auto mode, push delivered. `/home/uqaste15` marked trusted in `~/.claude.json` for this.
521 train (29915718) given `nice=100` on 2026-10-16 so 511/512/513 run first; 521 trains last (expected to finish ~Mon 19 Oct).

## Mid-run check, 2026-10-17 07:00 (job 29940932): healthy

- **Jobs.** 501 folds 0–3 COMPLETED (each about 8 h, `checkpoint_final` present).
  Running: 501 fold 4 (about epoch 545) and 511 folds 0, 1, 2 (about epochs 482, 472, 447).
  Pending on the per-user limit: 511 folds 3–4, and all of 512, 513 and 521.
  All exports and the main_end/final_end checks are pending on their dependencies.
- **Errors.** The `.err` files are empty, apart from the expected cancellation of
  29915715. There are no NaNs in any training log.
- **Losses.** The train loss falls steadily in every fold. Values at the last epoch:

  | Run | Train loss | Val loss | Marker pseudo-Dice |
  |---|---|---|---|
  | 501 folds 0–3 | about −0.94 | −0.49 to −0.64 | 0.49–0.56 |
  | 511 folds 0–2 | about −0.82 | about −0.72 to −0.74 | 0.73–0.75 |

  The 511 folds are clearly above the 501 folds at the same epoch. Validation
  plateaued early (see Needs Ashley).
- **Actions.** None needed: no resubmissions, no extra checks queued. The
  scheduled main_end check (29940933) and final_end check (29940934) still stand.
