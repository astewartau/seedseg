#!/usr/bin/env bash
# ONNX pipeline on CPU, one fold per array task: tasks 0-4 = folds 0-4 with mirroring,
# 5-9 = folds 0-4 without. Writes $OUT/cpu_{mirror,nomirror}/fold{k}/{sid}.nii.gz.
# Usage (from the repo root):
#   sbatch --array=0-9 --export=ALL,MODELS=<onnx dir>,INPUTS=<dir>,SUBJECTS=<file>,OUT=<dir> scripts/deploy/parity_cpu.slurm
#SBATCH --job-name=seedseg-parity-cpu
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --account=a_ai_collab
#SBATCH --output=/scratch/user/uqaste15/prostate-v2/deploy/logs/%x_%A_%a.out
set -euo pipefail
: "${MODELS:?}" "${INPUTS:?}" "${SUBJECTS:?}" "${OUT:?}"
T=${SLURM_ARRAY_TASK_ID:-0}; F=$((T % 5)); PREC=${PREC:-fp32}
if [ "$T" -lt 5 ]; then M=mirror; FLAG=; else M=nomirror; FLAG=--no-mirror; fi
EXTRA=; [ "$PREC" != fp32 ] && EXTRA="--precision $PREC" && M="${M}_$PREC"
/scratch/user/uqaste15/envs/seedseg-deploy/bin/python "${SLURM_SUBMIT_DIR:-.}/scripts/deploy/run_parity.py" \
    --models "$MODELS" --inputs "$INPUTS" --subjects "$SUBJECTS" --out "$OUT/cpu_$M" --folds $F $FLAG $EXTRA
