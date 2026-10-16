You are a scheduled mid-run health check for the SeedSeg v2 prostate project. Read `/scratch/user/uqaste15/src/seedseg/V2_PLAN.md` first. It has the plan, the dataset IDs and the job IDs.

Task: the main nnU-Net run (dataset 511, corrected labels) started about 24 h ago. Check that training is healthy across all running nnU-Net datasets (501, 511, 512, 513, 521):
- job states (`sacct` / `squeue`);
- errors in `/scratch/user/uqaste15/prostate-v2/logs/nnunet_train_*`;
- for each running fold, the training log under `nnUNet_results/<dataset>/*/fold_*/training_log_*.txt`: epochs completed, epoch time, train and validation loss falling, pseudo-Dice for the marker class rising, no NaNs.

Also check whether the GPU smoke test (29913696) ran and what its epoch time was. From the measured epoch times, update the ETAs in `V2_PLAN.md`.

This is a light check. If everything is healthy, update the plan and notify; don't start any analysis. If a fold has crashed, follow the retry rule below.

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
