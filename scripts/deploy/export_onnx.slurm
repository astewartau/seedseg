#!/usr/bin/env bash
# Export the deployment nnU-Net folds to ONNX (fp32 + fp16) with an ORT-vs-PyTorch check.
# Usage: sbatch --export=ALL,MODEL=<trainer__plans__3d_fullres dir>,OUT=<dir> scripts/deploy/export_onnx.slurm
#SBATCH --job-name=seedseg-onnx
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --account=a_ai_collab
#SBATCH --output=/scratch/user/uqaste15/prostate-v2/deploy/logs/%x_%j.out
set -euo pipefail
: "${MODEL:?}" "${OUT:?}"
source /scratch/user/uqaste15/src/seedseg-nnunet/scripts/nnunet/env.sh
export PATH=/scratch/user/uqaste15/envs/seedseg-deploy/bin:$PATH
python "${SLURM_SUBMIT_DIR:-.}/scripts/deploy/export_onnx.py" --model "$MODEL" --folds 0 1 2 3 4 --out "$OUT" --fp16 --check
