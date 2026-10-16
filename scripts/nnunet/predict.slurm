#!/usr/bin/env bash
# After all folds are trained: export probability maps as NIfTI.
#  - out-of-fold (OOF) marker/prostate probabilities for every training case,
#    taken from fold_X/validation/*.npz written by `nnUNetv2_train --npz`
#  - if the dataset has imagesTs (Design B): 5-fold ensemble prediction with
#    --save_probabilities, then exported likewise.
# Usage: sbatch --export=ALL,DATASET=502 scripts/nnunet/predict.slurm
#   optional: TRAINER, PLANS, FOLDS="0 1 2 3 4"
#SBATCH --job-name=nnunet-pred
#SBATCH --partition=gpu_cuda
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --account=a_ai_collab
#SBATCH --output=/scratch/user/uqaste15/prostate-v2/logs/nnunet_pred_%j.out
#SBATCH --error=/scratch/user/uqaste15/prostate-v2/logs/nnunet_pred_%j.err
set -euo pipefail
: "${DATASET:?set DATASET=<id>}"
TRAINER=${TRAINER:-nnUNetTrainer}
PLANS=${PLANS:-nnUNetResEncUNetMPlans}
FOLDS=${FOLDS:-0 1 2 3 4}
source /scratch/user/uqaste15/src/seedseg-nnunet/scripts/nnunet/env.sh
name=$(basename "$(ls -d "$nnUNet_raw"/Dataset"$(printf %03d "$DATASET")"_*)")
model="$nnUNet_results/$name/${TRAINER}__${PLANS}__3d_fullres"
out="$NNUNET_BASE/predictions/$name/${TRAINER}__${PLANS}"
mkdir -p "$out"
echo "Dataset $name, model $model -> $out at $(date)"

python "$SEEDSEG_REPO/scripts/nnunet/export_probs.py" oof --model "$model" \
    --raw "$nnUNet_raw/$name" --folds $FOLDS --out "$out/oof"

if [ -d "$nnUNet_raw/$name/imagesTs" ]; then
    nnUNetv2_predict -i "$nnUNet_raw/$name/imagesTs" -o "$out/test_ensemble_raw" \
        -d "$DATASET" -c 3d_fullres -tr "$TRAINER" -p "$PLANS" -f $FOLDS --save_probabilities
    python "$SEEDSEG_REPO/scripts/nnunet/export_probs.py" dir --pred "$out/test_ensemble_raw" \
        --images "$nnUNet_raw/$name/imagesTs" --out "$out/test_ensemble"
fi
echo "Done at $(date)"
