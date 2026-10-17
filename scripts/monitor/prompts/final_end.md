You are a scheduled end-of-run check for the SeedSeg v2 prostate project. Read `/scratch/user/uqaste15/src/seedseg/V2_PLAN.md` first, especially "How to resume" and any results an earlier check added.

Trigger: the training arrays for datasets **512** (corrected labels, temporal split), **513** (corrected + synthetic markers) and **521** (original-position spheres, the shape control) have ended. Tasks:
1. **Verify training and exports** as in the main check: all folds have `checkpoint_final.pth`, failed folds resubmitted once, export jobs done (or run them), maps complete and correctly oriented. For 512, also check the 5-fold ensemble predictions for the 25 test subjects.
2. **Evaluate** with `evaluate_detections.py` against the corrected references:
   - 513 and 521 on all 127 subjects and on the 25 temporal test subjects;
   - 512 on its 25 held-out test subjects, the direct comparison with the preprint's design and the released-model baseline;
   - the gradient-boosting re-ranker gave no gain on 511, so skip it;
   - **constellation selection** (`scripts/rerank/constellation.py`, added 2026-10-17) on 513's and 521's out-of-fold candidates:
     - use the same pre-specified settings as for 511 (inner-CV λ, K = reference count);
     - also report the extra-candidates variant, clearly labelled as not chosen blind;
     - compare with 511 + constellation (356/377, 7 FP; see "Follow-up analyses" in the plan);
   - use `scripts/evaluation/compare_arms.py` for paired comparisons and miss categories.
2b. **Repeat 250-epoch runs of 511**, if Ashley submitted them:
   - they are `TRAINER=nnUNetTrainer_250epochs` with `RESULTS_TAG=r1/r2/r3`, so their results are in `nnUNet_results_r*` and `predictions_r*`;
   - check with `sacct` (job name nnunet-train) and `ls /scratch/user/uqaste15/prostate-v2/nnunet/nnUNet_results_r*/*/*/fold_*/checkpoint_final.pth`;
   - if all 3×5 folds and their exports are done, evaluate:
     - each run alone;
     - the average of the 3 runs' out-of-fold maps (same splits, so still out-of-fold);
     - that average plus constellation selection;

     against 511 at 1000 epochs. This tests "repeat runs + selection" and "shorter schedule".
   - If they are still running, queue one follow-up check after them (see Waiting).
   - If they were never submitted, say so; don't submit them yourself.
3. **Write up:** add to the results table in `V2_PLAN.md`:
   - 513 vs 511: does the synthetic data help, overall and on markers next to T1-bright blood? `analysis/hemorrhage.csv`-style flags can be recomputed with the bright-fraction definition in the plan's history, or simply report per-subject differences;
   - 521 vs 501 and 511: label shape versus label position;
   - 512 vs the released models on the temporal test.

   - the 250-epoch repeat runs and their average, if available (2b).

   Give confidence intervals and stay factual. Finish with a short "Recommended final model and next steps" paragraph for Ashley, covering:
   - which arm and decoding to deploy (including the count flag for review);
   - whether a CNN verifier or slice-profile refinement seems worthwhile;
   - adding the 2025 T1 patients (listed in the plan);
   - how to update the web app and OpenRecon.

   QSM/phase was tried earlier and set aside by Ashley: do not recommend it.
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
