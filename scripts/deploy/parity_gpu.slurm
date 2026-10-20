#!/usr/bin/env bash
# ONNX pipeline on GPU: parity maps for the given subjects (mirroring on and off, 5 folds)
# and a timing benchmark. Usage (from the repo root):
#   sbatch --export=ALL,MODELS=<onnx dir>,INPUTS=<dir>,SUBJECTS=<file>,OUT=<dir>,RAW=<raw T1> scripts/deploy/parity_gpu.slurm
#SBATCH --job-name=seedseg-parity-gpu
#SBATCH --partition=gpu_cuda
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --account=a_ai_collab
#SBATCH --output=/scratch/user/uqaste15/prostate-v2/deploy/logs/%x_%j.out
set -euo pipefail
: "${MODELS:?}" "${INPUTS:?}" "${SUBJECTS:?}" "${OUT:?}" "${RAW:?}"
PY=/scratch/user/uqaste15/envs/seedseg-deploy-gpu/bin/python
S="${SLURM_SUBMIT_DIR:-.}/scripts/deploy"
nvidia-smi -L
$PY "$S/bench.py" --models "$MODELS" --input "$RAW" --device cuda --five-mirror
$PY "$S/run_parity.py" --models "$MODELS" --inputs "$INPUTS" --subjects "$SUBJECTS" --out "$OUT/gpu_mirror" --device cuda
$PY "$S/run_parity.py" --models "$MODELS" --inputs "$INPUTS" --subjects "$SUBJECTS" --out "$OUT/gpu_nomirror" --device cuda --no-mirror
$PY "$S/bench.py" --models "$MODELS" --input "$RAW" --device cuda --five-mirror --precision fp16 || true
echo "Done at $(date)"
