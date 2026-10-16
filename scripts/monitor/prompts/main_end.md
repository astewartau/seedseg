You are a scheduled end-of-run check for the SeedSeg v2 prostate project. Read `/scratch/user/uqaste15/src/seedseg/V2_PLAN.md` first, especially "How to resume". It has the plan, dataset IDs, job IDs and commands.

Trigger: the training arrays for dataset **511** (corrected labels, the main result) and **501** (original labels) have ended. That could mean finished, failed or cancelled. Tasks:
1. **Verify training:** for every fold 0–4 of 511 and 501, check that `checkpoint_final.pth` exists and the job ended COMPLETED. For a failed or timed-out fold, find the cause in its log and resubmit once (see rules).
2. **Verify exports:** if all folds are complete, the probability-export jobs (see `V2_PLAN.md`) should have started automatically; wait for them if they are still running. If an export didn't run or failed, run it with `predict.slurm` as the plan describes. Then check that every subject has a marker-probability map, and spot-check one map for orientation against its image and label. The export script prints a warning on argmax/seg mismatch; look for it in the log.
3. **Evaluate:** run `scripts/evaluation/evaluate_detections.py` against the corrected references for both 511 and 501 (all 127 subjects, plus the 25 temporal test subjects in `/scratch/user/uqaste15/prostate-v2/eval/test25.txt`), and also each single fold's model where available. Then run the re-ranker on the 511 out-of-fold candidates and evaluate it with `--rescore`. Use the Python environment noted in the plan.
4. **Write up:** add a results table to `V2_PLAN.md` comparing these against the released-model baseline already in the plan:
   - 511 vs 501 (the effect of the label fix), with confidence intervals;
   - the re-ranker;
   - single fold vs ensemble, where applicable.

   Also summarise the remaining errors, e.g. how many misses had no candidate versus were ranking errors. Be factual and don't overclaim, especially on the 25-subject test set.
5. Run `scripts/backup_v2_to_rdm.sh`, which now also captures the trained models.

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
