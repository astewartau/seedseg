# SeedSeg v2: plan and status

The goal is to improve gold-marker detection in prostate T1 MRI beyond the
preprint (bioRxiv 10.64898/2026.06.18.733061). The approach has four parts:
fix the reference labels, retrain with nnU-Net, add physics-simulated markers,
and evaluate with cross-validation over all 127 subjects.

Last updated: 2026-10-17 ~21:30 (main_end session, follow-up analyses requested by Ashley).

## Needs Ashley

- **Submit the 3× 250-epoch 511 runs** (auto mode blocked my sbatch). Also: why were the 2025 T1+RTSTRUCT
  patients left out of the 127? (z1438488 is answered: the original data marks it
  `EXCLUDE_INVALID_SEGMENTATION`, a label-derivation failure that the v2 DICOM pipeline might now
  fix. It's one subject, so low priority. z0354148 and z0791295 were excluded for missing T1.)
  **QSM: deprioritised (Ashley, 2026-10-17).** It was investigated earlier on a QSM dataset of about
  30 subjects on the RDM and never beat T1, and the project has moved on. Don't propose a QSM pilot again.
- **511 misses 15 clear voids outright, and 8 of them are found by 501 (main_end check, 2026-10-17).**
  The label fix does not help on test25: 511 gets 67/74, 501 gets 71/74. Of 511's 7 test misses,
  5 are markers with marker probability below 0.1 everywhere within 5 mm, and 4 of those 5 are detected
  at p ≈ 1 by 501 (z0008611 m2, z0324733 m0, z1623770 m2, z6031238 m0). All 15 of these markers have
  void strength 0.43–0.75 and are correctly labelled in 511 `labelsTr` (about 115 marker voxels each).
  Each is in a 3-marker subject where 511 output only 2 components. The misses are spread over all
  5 folds. Many of them (not all) lie at or just outside the CTV edge. Seven of the 15 are also missed
  by 501, so there is a shared hard set. Montage: `prostate-v2/eval/analysis/misses_511_nocand.png`.
  Table: `misses_511_501.csv`. I haven't changed anything. This needs a proper look before 511 is
  declared the main model: check whether the corrected CTV/marker rasterisation teaches "dark void at
  the prostate edge = prostate/background", and look at these voids by eye.
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
| 3 | nnU-Net v2 ResEnc-M, 5-fold CV | **501 and 511 done and evaluated** (2026-10-17; see results below); 512/513/521 training | `scripts/nnunet/` in worktree `seedseg-nnunet`, branch `v2-nnunet` |
| 4 | Physics-based synthetic markers and distractors | **Done and validated**; 254 cases generated | `scripts/synth/` (README = methods), worktree `seedseg-synth`, branch `v2-synth` |
| 5 | Second-stage candidate re-ranker | **Evaluated on 511 OOF candidates: no net gain** (all 127: +2/−2 markers, FP 9→9) | `scripts/rerank/rerank.py` |
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

## Results: 511 vs 501 (main_end check, 2026-10-17)

The nnU-Net predictions are out-of-fold (OOF): each subject is predicted only by the fold model that
did not train on it. All evaluation is against the corrected references, using 5 mm one-to-one
matching, threshold 0.1, top 3 per subject by mean probability. The CIs are 95% Clopper-Pearson.
"Released" is the 4-model ONNX ensemble. Its all-127 row is **partly in-sample** (the old models
trained on most of these subjects), so that comparison is biased in the released models' favour.

| Model | Set | Sensitivity (95% CI) | Precision (95% CI) | FP | Perfect subjects |
|---|---|---|---|---|---|
| Released ensemble | all 127 | 342/377 = 90.7% (87.3–93.4) | 342/377 = 90.7% (87.3–93.4) | 35 | 93/127 |
| 501 nnU-Net, original labels | all 127 | 351/377 = 93.1% (90.1–95.4) | 351/367 = 95.6% (93.0–97.5) | 16 | 102/127 |
| **511 nnU-Net, corrected labels** | all 127 | **354/377 = 93.9% (91.0–96.1)** | **354/363 = 97.5% (95.3–98.9)** | **9** | **103/127** |
| 511 + re-ranker | all 127 | 354/377 = 93.9% (91.0–96.1) | 354/363 = 97.5% (95.3–98.9) | 9 | 103/127 |
| Released ensemble | test 25 | 70/74 = 94.6% (86.7–98.5) | 70/75 = 93.3% (85.1–97.8) | 5 | 20/25 |
| 501 | test 25 | 71/74 = 95.9% (88.6–99.2) | 71/74 = 95.9% (88.6–99.2) | 3 | 23/25 |
| 511 | test 25 | 67/74 = 90.5% (81.5–96.1) | 67/69 = 97.1% (89.9–99.6) | 2 | 18/25 |
| 511 + re-ranker | test 25 | 68/74 = 91.9% (83.2–97.0) | 68/69 = 98.6% (92.2–100) | 1 | 19/25 |

The test-25 rows for 501/511 are OOF predictions from the CV models. They are not a temporal split:
these models trained on later-acquired subjects too. 512 will provide the temporal comparison.

**Paired comparisons** (same subjects, per-marker; subject-bootstrap 95% CI of the difference; exact McNemar on discordant markers):

| Contrast | Set | Markers found only by A / only by B | ΔSensitivity (95% CI) | ΔPrecision (95% CI) | McNemar p |
|---|---|---|---|---|---|
| 511 vs 501 (label fix) | all 127 | 15 / 12 | +0.8 pp (−1.6 to +3.2) | +1.9 pp (−0.1 to +4.0) | 0.70 |
| 511 vs 501 | test 25 | 1 / 5 | −5.4 pp (−12.0 to 0.0) | +1.2 pp (−3.1 to +6.2) | 0.22 |
| 511 vs released | all 127 (biased toward released) | 23 / 11 | +3.2 pp (+0.3 to +6.2) | +6.8 pp (+4.1 to +9.7) | – |
| 511 vs released | test 25 | 2 / 5 | −4.1 pp (−10.8 to +2.7) | +3.8 pp (−1.7 to +10.3) | – |
| 511 re-ranked vs 511 | all 127 | 2 / 2 | 0 (−0.8 to +0.8) | 0 | – |

**Single fold vs ensemble.** In 5-fold CV, each subject has exactly one held-out model, so the
5-model ensemble can't be evaluated on these subjects without leakage. That comparison waits for 512
(temporal test with the 5-fold ensemble, export job 29934250). Per-fold results, where each single
model is evaluated on its own held-out subjects:

| Fold (n subjects) | 501 sens | 501 prec | 511 sens | 511 prec | 511 perfect |
|---|---|---|---|---|---|
| 0 (26) | 71/76 | 71/74 | 71/76 | 71/74 | 21/26 |
| 1 (26) | 72/78 | 72/76 | 73/78 | 73/73 | 21/26 |
| 2 (25) | 69/74 | 69/73 | 71/74 | 71/74 | 21/25 |
| 3 (25) | 71/74 | 71/72 | 71/74 | 71/72 | 22/25 |
| 4 (25) | 68/75 | 68/72 | 68/75 | 68/70 | 18/25 |

**What this does and doesn't show.**
- Both nnU-Net arms beat the released ensemble on all 127, mostly through fewer false positives
  (35 → 16 → 9). This holds despite the bias in the released models' favour.
- 511 vs 501: 511 has fewer FPs (9 vs 16) and 3 more TPs on all 127. Neither difference is
  significant, since the CIs include 0. On test 25, 511 is 4 markers *worse* than 501. With n=25 that
  isn't significant either (p=0.22), but the direction is the opposite of the hypothesis. **The label
  fix has not been shown to improve detection.** Its clearest effect is precision.
- The re-ranker adds nothing on all 127. There are only about 20 non-marker candidates per training
  fold, so it has little to learn from. The +1 on test 25 is one marker.

**Remaining errors, 511 (all 127): 23 misses, 9 FPs.**
- 15 misses have **no candidate**: marker probability below 0.1 everywhere within 5 mm. All are clear
  voids in 3-marker subjects where 511 output only 2 components (see Needs Ashley). 7 of them are also
  missed by 501 and 5 by the released models.
- 8 misses are **ranking errors**: a candidate exists but a look-alike outranks it, so it falls out of
  the top 3. These account for 7 of the 9 FPs.
- The candidate recall ceiling is 95.8% (361/377), the same as 501's.
- 501 for comparison: 26 misses = 11 ranking + 15 "no candidate". 6 of those 15 have p ≈ 1 nearby
  but a merged or displaced component whose centroid is more than 5 mm away; 9 are truly unsegmented.
  The released models: 22 ranking + 13 no candidate.

Files: `prostate-v2/eval/results/nnunet{501,511}_{all127,test25,fold0..4}_*`,
`nnunet511_rerank*.csv`, `nnunet511_reranked_*`; analysis `prostate-v2/eval/analysis/`
(`markers_*.csv`, `misses_511_501.csv`, montage). Scripts: `scripts/evaluation/run_eval_cv.sh`
(all/test/per-fold) and `scripts/evaluation/compare_arms.py` (paired comparison and miss categories).

## Follow-up analyses (2026-10-17 evening, requested by Ashley): how to go further

All analyses below are CPU-only on the existing 501/511 out-of-fold outputs and are evaluated on all 127
subjects (plain 511: 354/377, 9 FP, 103 perfect).

| Variant (all 127) | Sensitivity | FP | Perfect | Notes |
|---|---|---|---|---|
| 511 plain | 354/377 | 9 | 103 | – |
| Average of 501 + 511 probability maps | 357/377 | 13 | 107 | paired vs 511: +9/−6 markers, ΔSens CI −1.1 to +2.7 pp |
| 511 + constellation selection (pre-specified) | 356/377 | 7 | 105 | +5/−3, ΔSens CI −0.5 to +1.6 pp |
| 511 + constellation + extra candidates p>1e-3 | 359/377 | 8 | 108 | threshold **not** chosen blind |
| 511 + constellation + extra candidates p>1e-4 | 361/377 | 8 | 110 | +10/−3, CI +0.5 to +3.4 pp; threshold informed by looking at the misses, so optimistic |

Test 25: average 70/74 with 4 FP; constellation 68/74 with 1 FP; constellation plus extras at 1e-4 69/74 with 2 FP.

- **Averaging** recovers 7 of the 8 markers that 501 finds and 511 misses. It loses 6 that only 511
  found, because detections from a single model drop to p ≈ 0.5 and are then outranked. Repeat runs
  of the *same* model plus better selection should do better than mixing arms. This needs the 3×
  250-epoch 511 runs (not yet submitted: the Slurm submission was blocked by the auto-mode
  classifier, so Ashley should run the command recorded in the session, using
  `RESULTS_TAG=r1..r3`, `TRAINER=nnUNetTrainer_250epochs`).
- **Constellation selection** (`scripts/rerank/constellation.py`) chooses the K candidates that
  jointly maximise candidate score plus a geometric prior learned from the training folds. K is the
  reference marker count, standing in for the implant record. λ was chosen by inner CV within each
  outer fold; every fold picked 0.1, so the prior acts as a light tie-breaker. It fixes 5 ranking
  swaps and loses 3.
- **Count flag.** If the implant count is known, subjects where fewer than K markers are output flag
  themselves for review, with no false alarms. For plain 511 that is 15 subjects (all 127).
- **The 11 markers that both 501 and 511 miss, looked at at full resolution**
  (`eval/analysis/hardcore_11.png`):
  - 4 are ranking errors: 511 has a strong blob on a clear void (z0147288, z2514519, z3137578, z1982224).
  - 3 are clear voids that the networks ignore, at vessel junctions or fat-plane edges (z1818331, z3284359, z0152238).
  - 3 have difficult surroundings: abutting T1-bright blood (z1451132, z1219334) or at the edge of rectal air (z6029265).
  - 1 has no distinct void visible (z1848016; the refinement did not move it).
- **Phase / QSM data exists, but not for the 127.** All 127 study subjects are magnitude-only T1
  (3T). **12 other patients (2022-11 to 2023-07) have a 7-echo QSM acquisition**
  (`t2starME_qsm_tra_Iso1.4mm_INPHASE_bipolar`, with magnitude, phase and scanner QSM):
  - 9 also have CT; 5 also have the T1 seed sequence;
  - none has REG or RTSTRUCT, so CT→MR registration would be needed for labels.

  Also available: 1 patient with iSWI phase (z0381949) and the gold-seed phantom (4-echo QSM).
  This is enough for a pilot on whether QSM separates gold from calcification and blood in vivo.
  Inventory: `eval/analysis/dicom_series_inventory.txt`.
- **Possible extra T1 training subjects not in the 127** (reason for exclusion unknown):
  - 2025 patients with T1 and RTSTRUCT: z2153014, z3152040, z1224615, z2743361 (+REG), z2969010 (+REG);
  - 2025 OpenRecon test patients with T1 (and CT for some);
  - z1438488 (2021, with CT, MR, REG and RTSTRUCT).

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

## Main-end check, 2026-10-17 19:18 (job 29940933): training OK, result mixed

- **Training.** All 10 folds of 501 (29915716) and 511 (29929993) COMPLETED in 7 h 53 m to 8 h 18 m,
  each with `checkpoint_final.pth`. There were no resubmissions.
- **Export.** 501 (29934246) and 511 (29934247) COMPLETED. Each wrote 127 OOF cases with no
  argmax/seg warnings. Orientation was spot-checked on z0008611, z0015061, z0099570 (501) and z3154324
  (511): affines match the image, marker probability is high inside the corrected marker spheres, and
  the markers sit on T1 voids. For z3154324, 511 gives p=0.68 in the corrected label and 0.00 in the
  displaced original label.
- **Evaluation and re-ranker:** see "Results: 511 vs 501" above, and Needs Ashley.
- **Still running:** 512 folds 0–2 (3–4 pending), then 513 and 521. The final_end check (29940934)
  still stands. No extra check was queued.
- The RDM backup was re-run after this check (it includes the 501/511 models).
