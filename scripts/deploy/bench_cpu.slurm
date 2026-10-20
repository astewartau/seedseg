#!/usr/bin/env bash
# CPU timing of the ONNX pipeline on one raw T1 (N4 + 1/5 folds, mirroring on/off).
# Usage: sbatch --cpus-per-task=<n> --export=ALL,MODELS=<dir>,RAW=<raw T1> scripts/deploy/bench_cpu.slurm
#SBATCH --job-name=seedseg-bench-cpu
#SBATCH --partition=general
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:30:00
#SBATCH --account=a_ai_collab
#SBATCH --output=/scratch/user/uqaste15/prostate-v2/deploy/logs/%x_%j.out
set -euo pipefail
lscpu | grep -E "Model name|^CPU\(s\)"
/scratch/user/uqaste15/envs/seedseg-deploy/bin/python "${SLURM_SUBMIT_DIR:-.}/scripts/deploy/bench.py" --models "$MODELS" --input "$RAW" --device cpu
