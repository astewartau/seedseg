#!/usr/bin/env bash
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --account=a_ai_collab
#SBATCH --output=logs/eval_ablation_%j.out
#SBATCH --error=logs/eval_ablation_%j.err

export PYTHONUNBUFFERED=1

export CONDA_HOME=$HOME/miniconda3
source "$CONDA_HOME/etc/profile.d/conda.sh"
conda activate prostate39

echo "Starting model count ablation on $(hostname) at $(date)"
cd "$SLURM_SUBMIT_DIR"

python scripts/evaluation/evaluate_model_count_ablation.py

echo "Finished at $(date)"
