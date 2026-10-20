#!/bin/bash
#SBATCH --job-name=claude-resume
#SBATCH --account=a_ai_collab
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=16:00:00
#SBATCH --output=/scratch/user/uqaste15/prostate-v2/logs/claude_resume_%j.out
#
# Reopen an earlier Claude Code conversation on a fresh node, visible via Remote Control,
# e.g. queued behind the interactive job that hosts it:
#   sbatch --dependency=afterany:<jobid> --export=ALL,SESSION=<session-id>,RC_NAME=<name> \
#          scripts/monitor/resume_session.slurm
# PROMPT (optional) is sent as the first message after resuming.
set -u
export PATH="$HOME/.nvm/versions/node/v24.13.1/bin:$HOME/.local/bin:$PATH"
: "${SESSION:?set SESSION=<session-id>}"
RC_NAME=${RC_NAME:-resume-$(date +%m%d-%H%M)}
cd "$HOME"
P=/scratch/user/uqaste15/prostate-v2/logs/claude_resume_${SLURM_JOB_ID}_prompt.txt
printf '%s' "${PROMPT:-Resumed on a new node after the previous Bunya job ended. Continue where we left off.}

(Resumed by Slurm job $SLURM_JOB_ID on $(hostname) at $(date), wall time ${SLURM_TIMELIMIT:-16h}.)" > "$P"
# private tmux server per job (a shared default socket on the node is one suspected cause of an
# earlier check exiting instantly); claude's stderr goes to the job log directory
TM=(tmux -S "/tmp/claude-resume-$SLURM_JOB_ID.sock")
ERR=/scratch/user/uqaste15/prostate-v2/logs/claude_resume_${SLURM_JOB_ID}.err
"${TM[@]}" new-session -d -s "$RC_NAME" \
  "claude --resume '$SESSION' --remote-control '$RC_NAME' --permission-mode auto \"\$(cat $P)\" 2>>$ERR; echo \"claude exited \$? at \$(date)\" >>$ERR; sleep 120"
echo "started tmux session $RC_NAME at $(date) (socket /tmp/claude-resume-$SLURM_JOB_ID.sock)"
sleep 5; "${TM[@]}" has-session -t "$RC_NAME" 2>/dev/null || echo "WARNING: tmux session gone after 5 s; see $ERR"
while "${TM[@]}" has-session -t "$RC_NAME" 2>/dev/null; do sleep 60; done
echo "session ended at $(date)"
