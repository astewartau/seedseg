#!/usr/bin/env bash
# Write fp16-weight ("w16") copies of the exported ONNX folds. Usage: sbatch --export=ALL,MODELS=<dir> scripts/deploy/make_w16.slurm
#SBATCH --job-name=seedseg-w16
#SBATCH --partition=general
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --account=a_ai_collab
#SBATCH --output=/scratch/user/uqaste15/prostate-v2/deploy/logs/%x_%j.out
set -euo pipefail
/scratch/user/uqaste15/envs/seedseg-deploy/bin/python "${SLURM_SUBMIT_DIR:-.}/scripts/deploy/make_w16.py" "$MODELS"
