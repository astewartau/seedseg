#!/usr/bin/env bash
# Mirror the SeedSeg v2 work from scratch to an organised tree on the QSMFUNCTOR RDM (Q0748):
#   /QRISdata/Q0748/data/prostate/v2/   (layout and provenance: scripts/rdm_README.md -> README.md there)
# Incremental (rsync), never deletes anything on the RDM; re-run any time (e.g. after new models).
# Large regenerable intermediates are deliberately not copied; README.md says how to rebuild them.
set -euo pipefail
R=/scratch/user/uqaste15/prostate-v2
SRC=/scratch/user/uqaste15/src
REPO=$SRC/seedseg
DEST=${DEST:-/QRISdata/Q0748/data/prostate/v2}
# Released preprint ONNX models: already in $DEST/models/released_onnx (and on OSF f7hjv); set ONNX=<dir> only to re-copy them.
ONNX=${ONNX:-}
RS=(rsync -rlt --chmod=ug+rwX,o-rwx)   # no owner/group/perms: the RDM enforces its own group (Q0748RW)
mkdir -p "$DEST"/{docs,code,data,models,predictions,evaluation,deploy,archive}
cd "$R"

# docs
cp "$REPO/V2_PLAN.md" "$REPO/METHODS_V2.md" "$DEST/docs/"
cp "$REPO/scripts/labels/README.md" "$DEST/docs/labels_README.md"
cp "$SRC/seedseg-synth/scripts/synth/README.md" "$DEST/docs/synth_README.md"
cp "$SRC/seedseg-nnunet/scripts/nnunet/README.md" "$DEST/docs/nnunet_README.md"

# code: one bundle with every branch (v2-labels, v2-nnunet, v2-synth) and full history
git -C "$REPO" bundle create "$DEST/code/seedseg-v2.bundle.tmp" --all 2>/dev/null
mv "$DEST/code/seedseg-v2.bundle.tmp" "$DEST/code/seedseg-v2.bundle"
git -C "$REPO" for-each-ref --format='%(refname:short) %(objectname:short) %(subject)' refs/heads > "$DEST/code/branches.txt"

# data: study set (127), synthetic cases, external set
mkdir -p "$DEST/data/study"
cp subjects.txt splits_original.json eval/test25.txt eval/val9.txt "$DEST/data/study/"
[ -f eval/train102_temporal.txt ] && cp eval/train102_temporal.txt "$DEST/data/study/"
"${RS[@]}" -L images/ "$DEST/data/study/images/"            # N4-corrected T1 (symlinks dereferenced)
"${RS[@]}" labels/ "$DEST/data/study/labels/"
"${RS[@]}" labelfix/ "$DEST/data/study/labelfix/"
"${RS[@]}" synthetic/ "$DEST/data/synthetic/"
mkdir -p "$DEST/data/external"
cp fresh/subjects_fresh.txt "$DEST/data/external/"
"${RS[@]}" fresh/images/ "$DEST/data/external/images/"
"${RS[@]}" fresh/labels/ "$DEST/data/external/labels/"

# models: nnU-Net results (final checkpoints, plans, logs, validation segmentations and summaries;
# no checkpoint_best/latest or validation npz, no 5-epoch smoke test)
for d in nnunet/nnUNet_results nnunet/nnUNet_results_r*; do
  [ -d "$d" ] || continue
  "${RS[@]}" --prune-empty-dirs --exclude='nnUNetTrainer_5epochs__*/' \
    --exclude='checkpoint_best.pth' --exclude='checkpoint_latest.pth' --exclude='validation/*.npz' \
    "$d/" "$DEST/models/$(basename "$d")/"
done
"${RS[@]}" "$SRC/seedseg-nnunet/scripts/nnunet/splits/" "$DEST/models/splits/"
if [ -n "$ONNX" ] && [ -d "$ONNX" ]; then "${RS[@]}" "$ONNX/" "$DEST/models/released_onnx/"
else echo "released_onnx: no ONNX source given, keeping $DEST/models/released_onnx as is"; fi

# predictions: probability maps on the image grid
for d in nnunet/predictions nnunet/predictions_*; do
  [ -d "$d" ] || continue
  "${RS[@]}" "$d/" "$DEST/predictions/study_$(basename "$d")/"
done
"${RS[@]}" eval/released_probs/ "$DEST/predictions/study_released/"
[ -d eval/probs_avg ] && "${RS[@]}" eval/probs_avg/ "$DEST/predictions/study_avg501_511/"
# external set: symlinks (mprob/, 511r*_fold*) point at files synced here too; cache/ = constellation cache
"${RS[@]}" --no-links --exclude='cache/' fresh/pred/ "$DEST/predictions/external/"
for d in fresh/pred_*; do [ -d "$d" ] && "${RS[@]}" --no-links --exclude='cache/' "$d/" "$DEST/predictions/external_${d#fresh/pred_}/"; done

# evaluation
"${RS[@]}" eval/results/ "$DEST/evaluation/results/"
"${RS[@]}" eval/analysis/ "$DEST/evaluation/analysis/"
"${RS[@]}" --exclude='*.npy' --exclude='*.json' fresh/work/ "$DEST/evaluation/external_labelling_work/" 2>/dev/null || true
"${RS[@]}" logs/ "$DEST/evaluation/slurm_logs/"

# deploy: ONNX export, parity, browser/WebGPU, container and OpenRecon tests (hard links kept;
# container images, Apptainer caches, a pip env and browser profiles are not copied)
[ -d deploy ] && "${RS[@]}" -H --exclude='*.sif' --exclude='cache/' --exclude='container_bench/ortgpu_env/' \
  --exclude='webgpu_probe/tmp.*/' deploy/ "$DEST/deploy/"

# archive: record of the invalid first 521 run (labels without marker class); its models are not kept
# (the scratch copy was deleted on 2026-10-26 after this archive was made; skip when absent)
[ -d invalid_521_20261019 ] && "${RS[@]}" invalid_521_20261019/README.txt invalid_521_20261019/labels_original_sphere invalid_521_20261019/eval \
  "$DEST/archive/invalid_521_20261019/"

# README + inventory
{ cat "$REPO/scripts/rdm_README.md"
  echo; echo "## Inventory (generated $(date '+%Y-%m-%d %H:%M'))"; echo; echo '```'
  (cd "$DEST" && du -sh --apparent-size docs code data/* models/* predictions/* evaluation/* deploy/* archive/* 2>/dev/null)
  echo; echo "Models with checkpoint_final:"
  (cd "$DEST/models" && ls -d nnUNet_results*/*/*/fold_* 2>/dev/null | while read f; do if [ -f "$f/checkpoint_final.pth" ]; then echo "  $f"; fi; done) || true
  echo '```'; } > "$DEST/README.md"
echo "Synced to $DEST"; du -sh --apparent-size "$DEST" || true
