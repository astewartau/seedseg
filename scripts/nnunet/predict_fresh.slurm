#!/usr/bin/env bash
# Predict an external image folder with ONE fold model per array task (CPU), exporting
# probabilities on the original grid. Single-fold outputs can be averaged afterwards.
# Usage: sbatch --array=0-4 --export=ALL,DATASET=513,IN=<dir of {sid}_0000.nii.gz>,OUT=<dir> scripts/nnunet/predict_fresh.slurm
#SBATCH --job-name=nnunet-predfresh
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --account=a_ai_collab
#SBATCH --output=/scratch/user/uqaste15/prostate-v2/logs/nnunet_predfresh_%A_%a.out
#SBATCH --error=/scratch/user/uqaste15/prostate-v2/logs/nnunet_predfresh_%A_%a.err
set -euo pipefail
: "${DATASET:?}" "${IN:?}" "${OUT:?}"
FOLD=${SLURM_ARRAY_TASK_ID:-0}
TRAINER=${TRAINER:-nnUNetTrainer}; PLANS=${PLANS:-nnUNetResEncUNetMPlans}
source /scratch/user/uqaste15/src/seedseg-nnunet/scripts/nnunet/env.sh
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-24}
o="$OUT/${DATASET}_fold$FOLD"
nnUNetv2_predict -i "$IN" -o "$o/raw" -d "$DATASET" -c 3d_fullres -tr "$TRAINER" -p "$PLANS" -f "$FOLD" \
    --save_probabilities -device cpu
python "$SEEDSEG_REPO/scripts/nnunet/export_probs.py" dir --pred "$o/raw" --images "$IN" --out "$o/probs"
echo "Done at $(date)"
