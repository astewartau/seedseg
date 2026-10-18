You are a scheduled end-of-run check for the SeedSeg v2 prostate project. Read `/scratch/user/uqaste15/src/seedseg/V2_PLAN.md` first, especially "How to resume", "Results: 512, 513, 521" and the final-end check entry.

Trigger: the training arrays of two sets of jobs (submitted by Claude at Ashley's request on 2026-10-19) have ended. Their export jobs (CPU, about 3 min) may still be running or, if a fold failed, never start because of `afterok`; wait for them or run them yourself:
- **521 rerun** (shape-vs-position control, labels regenerated after the reshape-copy bug fix 901c54d):
  preprocess 30114402, train array 30114403, export 30114404. The invalid first run is kept in `prostate-v2/invalid_521_20261019/`; don't touch it.
- **3× 250-epoch repeats of 511** (`TRAINER=nnUNetTrainer_250epochs`, `RESULTS_TAG=r1/r2/r3`):
  train arrays 30114276 / 30114278 / 30114281, exports 30114277 / 30114280 / 30114282. Results are in
  `nnunet/nnUNet_results_r*`, predictions in `nnunet/predictions_r*/Dataset511_SeedSegCvAll_corrected/nnUNetTrainer_250epochs__nnUNetResEncUNetMPlans/oof`.

Tasks:
1. **Verify:** all 5 folds of 521 and 3×5 of the repeats have `checkpoint_final.pth`; failed folds resubmitted once (then re-run that export); exports complete, 127 OOF cases each, orientation spot-checked. Check that the 521 labels in `nnUNet_raw/Dataset521…/labelsTr` contain marker class 1 (126/127 should; z2877562 has none).
2. **Evaluate** against the corrected references, as for the other arms (`eval/run_eval.sh <id> <oof>`; use a distinct id per run, e.g. `521`, `511r1`, `511r2`, `511r3`):
   - 521 on all 127 and test 25; paired comparisons (`scripts/evaluation/compare_arms.py`) of 521 vs 501 and 521 vs 511, with miss categories and exact McNemar p. Interpretation: 501 → 521 = shape only (same positions, spheres); 521 → 511 = position/CTV correction.
   - each 250-epoch run alone; the **average of the 3 runs' marker-probability maps and full 3-class `_probs` maps** (same folds, so still out-of-fold; write averaged maps to a new dir and evaluate them); and that average + constellation selection (`scripts/rerank/constellation.py --oof <averaged probs dir> --cache <new per-arm cache> --folds <511 splits>`, pre-specified settings: inner-CV λ, K = implant count; extras variant only if clearly labelled as not chosen blind). Compare with 511 at 1000 epochs (plain and + constellation) and with 513 + constellation (360/377, 8 FP), the current best.
   - Also run constellation selection on 521 for completeness.
3. **Write up** in `V2_PLAN.md`: a results section "Results: 521 rerun and 250-epoch repeats" with CIs and paired tests, stay factual. Update the "Recommended final model" paragraph only if the evidence changes it (e.g. the 3-run average + constellation clearly beats 513 + constellation). QSM/phase is out of scope: do not recommend it.
4. Run `scripts/backup_v2_to_rdm.sh`.

## Rules for this unattended check (agreed with Ashley, 2026-10-16)
- **Allowed:**
  - inspect jobs and logs (`squeue`, `sacct`, `scontrol`, the files under `/scratch/user/uqaste15/prostate-v2/logs/`);
  - resubmit or cancel **this project's own** nnU-Net/export/claude-check Slurm jobs;
  - edit files and run scripts in the SeedSeg worktrees under `/scratch/user/uqaste15/src/seedseg*`;
  - commit locally;
  - update `V2_PLAN.md`;
  - run `scripts/backup_v2_to_rdm.sh`.
- **Retries:** at most ONE resubmission per failed fold. `train.slurm` resumes from `checkpoint_latest` automatically. If a fold fails twice, or for a reason that a resubmission won't fix (code or data error), diagnose it, write it up, and stop.
- **Never:**
  - delete data, labels, synthetic cases or trained models (`nnUNet_results`);
  - push to GitHub;
  - touch Slurm jobs that aren't this project's (e.g. qsmci arrays);
  - change the experimental design (labels, splits, arms, trainer settings).

  If something needs a judgement call, write the question at the top of `V2_PLAN.md` under "Needs Ashley" and send a push notification.
- **Waiting:** for anything you resubmit or that is still running, don't wait for days. Queue at most one follow-up check: `sbatch --dependency=afterany:<jobid> --export=ALL,CHECK=<same check name> /scratch/user/uqaste15/src/seedseg/scripts/monitor/claude_check.slurm`. Note its job ID in `V2_PLAN.md`. Short waits (minutes to an hour, e.g. an export job finishing) are fine.
- **When done:**
  - update the status in `V2_PLAN.md` and its copy `~/notes/prostate-seedseg-V2_PLAN.md`;
  - update the SeedSeg project memory with the outcome;
  - commit;
  - send ONE push notification (PushNotification tool) with a 1–2 line summary: OK, problem found and fixed, or needs Ashley;
    If the tool reports the notification was skipped because the terminal is active, run `sleep 180` in the background, wait for it, and send it once more.
  - then stop. The session stays open so Ashley can read it.
