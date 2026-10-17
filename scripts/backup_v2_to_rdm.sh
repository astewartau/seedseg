#!/usr/bin/env bash
# Back up SeedSeg v2 work from scratch to the QSMFUNCTOR RDM (Q0748).
# Copies what is not already on the RDM and not cheaply regenerable:
# labels, label-fix records/QC, synthetic data, evaluation results, splits,
# nnU-Net dataset.json/splits, trained nnU-Net models (checkpoint_final + plans,
# if present), and a git bundle of all branches. Re-run any time; each run
# writes a dated snapshot.
set -euo pipefail
R=/scratch/user/uqaste15/prostate-v2
REPO=/scratch/user/uqaste15/src/seedseg
DEST=/QRISdata/Q0748/data/prostate/v2-work
STAMP=$(date +%Y%m%d-%H%M%S)
mkdir -p "$DEST"
cd "$R"
files=(labels labelfix synthetic eval/results eval/analysis splits_original.json subjects.txt)
for f in nnunet/nnUNet_raw/*/dataset.json nnunet/nnUNet_preprocessed/*/splits_final.json \
         nnunet/nnUNet_preprocessed/*/nnUNetResEncUNetMPlans.json \
         nnunet/nnUNet_results*/*/*/fold_*/checkpoint_final.pth nnunet/nnUNet_results*/*/*/plans.json \
         nnunet/nnUNet_results*/*/*/dataset.json nnunet/nnUNet_results*/*/*/fold_*/training_log_*.txt; do
  [ -e "$f" ] && files+=("$f")
done
tar -cf "$DEST/v2-data-$STAMP.tar" "${files[@]}"
git -C "$REPO" bundle create "$DEST/seedseg-v2-$STAMP.bundle" --all
cp "$REPO/V2_PLAN.md" "$DEST/V2_PLAN-$STAMP.md"
ls -lh "$DEST" | tail -4
echo "Restore: tar -xf v2-data-$STAMP.tar -C /scratch/user/uqaste15/prostate-v2 ; git clone seedseg-v2-$STAMP.bundle seedseg"
