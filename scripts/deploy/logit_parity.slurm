#!/usr/bin/env bash
# Max |logit difference| of the WebGPU (fp32, fp16-stored) and int8 exports vs the original,
# one fold per array task (0-4). Usage (from the repo root):
#   sbatch --array=0-4 --export=ALL,MODELS=<onnx dir>,INT8=<int8 dir>,INPUTS=<dir>,SUBJECTS=<file>,OUT=<dir> scripts/deploy/logit_parity.slurm
#SBATCH --job-name=seedseg-logit-parity
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --account=a_ai_collab
#SBATCH --output=/scratch/user/uqaste15/prostate-v2/deploy/logs/%x_%A_%a.out
set -euo pipefail
: "${MODELS:?}" "${INT8:?}" "${INPUTS:?}" "${SUBJECTS:?}" "${OUT:?}"
F=${SLURM_ARRAY_TASK_ID:-0}
/scratch/user/uqaste15/envs/seedseg-deploy/bin/python "${SLURM_SUBMIT_DIR:-.}/scripts/deploy/logit_parity.py" \
    --models "$MODELS" --inputs "$INPUTS" --subjects "$SUBJECTS" --fold "$F" \
    --variant webgpu="$MODELS" --variant webgpu16="$MODELS" --variant int8="$INT8" --out "$OUT/logit_parity_fold$F.json"
