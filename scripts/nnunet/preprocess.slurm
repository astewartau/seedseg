#!/usr/bin/env bash
# Fingerprint + ResEnc-M planning + 3d_fullres preprocessing (CPU only).
# Usage: sbatch --export=ALL,DATASET=501 scripts/nnunet/preprocess.slurm
#SBATCH --job-name=nnunet-prep
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --account=a_ai_collab
#SBATCH --output=/scratch/user/uqaste15/prostate-v2/logs/nnunet_prep_%j.out
#SBATCH --error=/scratch/user/uqaste15/prostate-v2/logs/nnunet_prep_%j.err
set -euo pipefail
: "${DATASET:?set DATASET=<id>}"
source /scratch/user/uqaste15/src/seedseg-nnunet/scripts/nnunet/env.sh
echo "Preprocessing Dataset $DATASET on $(hostname) at $(date)"
nnUNetv2_plan_and_preprocess -d "$DATASET" -pl nnUNetPlannerResEncM -c 3d_fullres \
    --verify_dataset_integrity -np "${SLURM_CPUS_PER_TASK:-8}"
# (Re)install our fixed splits so nnU-Net never generates its own.
name=$(basename "$(ls -d "$nnUNet_raw"/Dataset"$(printf %03d "$DATASET")"_*)")
cp "$nnUNet_raw/$name/splits_final.json" "$nnUNet_preprocessed/$name/splits_final.json"
echo "Done at $(date)"
