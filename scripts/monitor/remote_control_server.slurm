#!/bin/bash
#SBATCH --job-name=claude-rc
#SBATCH --account=a_ai_collab
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/user/uqaste15/prostate-v2/logs/claude_rc_%j.out
#
# Run a Claude Code Remote Control server on a compute node, so new chats can be started
# from claude.ai/code or the mobile app (they run in $HOME on this node and share its CPUs).
#   sbatch [--export=ALL,RC_NAME=bunya] scripts/monitor/remote_control_server.slurm
set -u
export PATH="$HOME/.nvm/versions/node/v24.13.1/bin:$HOME/.local/bin:$PATH"
RC_NAME=${RC_NAME:-bunya}
cd "$HOME"
TM=(tmux -S "/tmp/claude-rc-$SLURM_JOB_ID.sock")
ERR=/scratch/user/uqaste15/prostate-v2/logs/claude_rc_${SLURM_JOB_ID}.err
"${TM[@]}" new-session -d -s "$RC_NAME" \
  "claude remote-control --name '$RC_NAME' --spawn same-dir --permission-mode auto 2>>$ERR; echo \"exited \$? at \$(date)\" >>$ERR; sleep 120"
echo "started Remote Control server $RC_NAME on $(hostname) at $(date)"
sleep 5; "${TM[@]}" has-session -t "$RC_NAME" 2>/dev/null || echo "WARNING: tmux session gone after 5 s; see $ERR"
while "${TM[@]}" has-session -t "$RC_NAME" 2>/dev/null; do sleep 60; done
echo "server ended at $(date)"
