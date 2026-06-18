#!/usr/bin/env bash
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --account=a_ai_collab
#SBATCH --output=logs/ablation_persubj_%j.out
#SBATCH --error=logs/ablation_persubj_%j.err

export PYTHONUNBUFFERED=1

export CONDA_HOME=$HOME/miniconda3
source "$CONDA_HOME/etc/profile.d/conda.sh"
conda activate prostate39

echo "Starting per-subject ablation (CPU) on $(hostname) at $(date)"
echo "CPUs: $SLURM_CPUS_PER_TASK"
cd "$SLURM_SUBMIT_DIR"

# Make PyTorch use all allocated CPUs
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

python scripts/evaluation/evaluate_model_count_ablation_per_subject.py

echo "Finished at $(date)"
