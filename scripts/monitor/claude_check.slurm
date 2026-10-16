#!/bin/bash
#SBATCH --job-name=claude-check
#SBATCH --account=a_ai_collab
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=16:00:00
#SBATCH --output=/scratch/user/uqaste15/prostate-v2/logs/claude_check_%j.out
#
# Start a Claude Code session, visible via Remote Control (claude.ai/code or the
# app), that works on a task prompt immediately: checking SeedSeg v2 Slurm jobs.
# Usage (submit with a Slurm dependency that fires at the right moment):
#   sbatch --dependency=afterany:<train array> --export=ALL,CHECK=main_end \
#          scripts/monitor/claude_check.slurm
# CHECK names a prompt in scripts/monitor/prompts/<CHECK>.md.
# The session stays open (for you to read or continue) until the wall time ends.
set -u
export PATH="$HOME/.nvm/versions/node/v24.13.1/bin:$HOME/.local/bin:$PATH"
PROMPTS=/scratch/user/uqaste15/src/seedseg/scripts/monitor/prompts
: "${CHECK:?set CHECK=<prompt name>}"
PROMPT_FILE="$PROMPTS/$CHECK.md"
[ -f "$PROMPT_FILE" ] || { echo "no prompt $PROMPT_FILE"; exit 1; }
NAME="seedseg-$CHECK-$(date +%m%d-%H%M)"
# Home directory so the user's CLAUDE.md and memory (incl. the SeedSeg project memory) load
cd "$HOME"
PROMPT="$(cat "$PROMPT_FILE")

(This session was started by Slurm job $SLURM_JOB_ID on $(hostname) at $(date), wall time ${SLURM_TIMELIMIT:-16h}.)"
printf '%s' "$PROMPT" > "/scratch/user/uqaste15/prostate-v2/logs/claude_check_${SLURM_JOB_ID}_prompt.txt"
# Claude's interactive UI needs a terminal: run it inside a detached tmux session
tmux new-session -d -s "$NAME" \
  "claude --remote-control '$NAME' --model claude-opus-5-5 --effort medium --permission-mode auto \"\$(cat /scratch/user/uqaste15/prostate-v2/logs/claude_check_${SLURM_JOB_ID}_prompt.txt)\"; sleep 60"
echo "started tmux session $NAME at $(date)"
while tmux has-session -t "$NAME" 2>/dev/null; do sleep 60; done
echo "session ended at $(date)"
