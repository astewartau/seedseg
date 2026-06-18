#!/bin/bash
# Train 4 production models with different random seeds, using ALL available data
# (93 training subjects + 25 previously-held-out test subjects = 118 subjects).
# The 9 validation subjects from data/val_subjects.txt remain the validation set.
# Run from project root: bash scripts/training/train_production_all_models.sh

mkdir -p models/production-all logs

for SEED in 42 123 456 789; do
    echo "Submitting production-all model with seed=${SEED}"
    sbatch --job-name="prodall-seed${SEED}" \
           --export=ALL,SEED=$SEED \
           scripts/training/train_production_all.slurm
done
