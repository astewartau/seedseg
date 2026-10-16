#!/usr/bin/env bash
# Train nnU-Net 3d_fullres ResEnc-M, one fold per array task.
# Usage: sbatch --array=0-4 --export=ALL,DATASET=501 scripts/nnunet/train.slurm
#   optional: TRAINER=nnUNetTrainer_5epochs (default nnUNetTrainer, 1000 epochs)
# --npz makes the final validation save softmax probabilities of the held-out
# fold (fold_X/validation/*.npz) -> out-of-fold probability maps.
# Resubmitting continues from the latest checkpoint (--c) if one exists.
#SBATCH --job-name=nnunet-train
#SBATCH --partition=gpu_cuda
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=96G
#SBATCH --time=72:00:00
#SBATCH --account=a_ai_collab
#SBATCH --output=/scratch/user/uqaste15/prostate-v2/logs/nnunet_train_%A_%a.out
#SBATCH --error=/scratch/user/uqaste15/prostate-v2/logs/nnunet_train_%A_%a.err
set -euo pipefail
: "${DATASET:?set DATASET=<id>}"
FOLD=${FOLD:-${SLURM_ARRAY_TASK_ID:-0}}
TRAINER=${TRAINER:-nnUNetTrainer}
PLANS=${PLANS:-nnUNetResEncUNetMPlans}
source /scratch/user/uqaste15/src/seedseg-nnunet/scripts/nnunet/env.sh
export nnUNet_n_proc_DA=${nnUNet_n_proc_DA:-$(( ${SLURM_CPUS_PER_TASK:-12} - 2 ))}
echo "Dataset $DATASET fold $FOLD trainer $TRAINER plans $PLANS on $(hostname) at $(date)"
nvidia-smi || { echo "No GPU available, exiting."; exit 1; }
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

name=$(basename "$(ls -d "$nnUNet_raw"/Dataset"$(printf %03d "$DATASET")"_*)")
outdir="$nnUNet_results/$name/${TRAINER}__${PLANS}__3d_fullres/fold_$FOLD"
extra=()
if [ -f "$outdir/checkpoint_final.pth" ]; then
    echo "checkpoint_final exists: running validation only"; extra=(--val)
elif [ -f "$outdir/checkpoint_latest.pth" ]; then
    echo "resuming from checkpoint_latest"; extra=(--c)
fi
nnUNetv2_train "$DATASET" 3d_fullres "$FOLD" -tr "$TRAINER" -p "$PLANS" --npz "${extra[@]}"
echo "Done at $(date)"
