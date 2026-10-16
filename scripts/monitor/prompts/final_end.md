You are a scheduled end-of-run check for the SeedSeg v2 prostate project. Read `/scratch/user/uqaste15/src/seedseg/V2_PLAN.md` first, especially "How to resume" and any results an earlier check added.

Trigger: the training arrays for datasets **512** (corrected labels, temporal split), **513** (corrected + synthetic markers) and **521** (original-position spheres, the shape control) have ended. Tasks:
1. **Verify training and exports** as in the main check: all folds have `checkpoint_final.pth`, failed folds resubmitted once, export jobs done (or run them), maps complete and correctly oriented. For 512, also check the 5-fold ensemble predictions for the 25 test subjects.
2. **Evaluate** with `evaluate_detections.py` against the corrected references:
   - 513 and 521 on all 127 subjects and on the 25 temporal test subjects;
   - 512 on its 25 held-out test subjects, the direct comparison with the preprint's design and the released-model baseline;
   - the re-ranker on 513's out-of-fold candidates, if 511's re-ranker helped.
3. **Write up:** add to the results table in `V2_PLAN.md`:
   - 513 vs 511: does the synthetic data help, overall and on markers next to T1-bright blood? `analysis/hemorrhage.csv`-style flags can be recomputed with the bright-fraction definition in the plan's history, or simply report per-subject differences;
   - 521 vs 501 and 511: label shape versus label position;
   - 512 vs the released models on the temporal test.

   Give confidence intervals and stay factual. Finish with a short "Recommended final model and next steps" paragraph for Ashley, covering which arm to deploy, whether a CNN re-ranker or slice-profile refinement seems worthwhile, and how to update the web app and OpenRecon.
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
