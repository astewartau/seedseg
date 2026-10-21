# nnU-Net v2 baselines for SeedSeg

nnU-Net v2 (`nnunetv2` 2.8.1, PyTorch 2.6.0 + CUDA 12.4, Python 3.11; conda env
`seedseg-nnunet`) trained on the prepared v2 data in
`<root>/` (images `images/{sid}.nii`, labels
`labels/{variant}/{sid}.nii.gz`, classes 0 background / 1 marker / 2 prostate).
nnU-Net folders: `<root>/nnunet/nnUNet_{raw,preprocessed,results}`.
Logs: `<root>/logs/`.

## Scripts

| File | Purpose |
|---|---|
| `env.sh` | Activates `seedseg-nnunet`, sets `nnUNet_raw/_preprocessed/_results`. |
| `convert.py` | Builds a raw dataset for a label variant and design; writes `dataset.json` and a fixed `splits_final.json` (also copied to `nnUNet_preprocessed` and `splits/`). `--synthetic` adds `synthetic/{variant}/` cases. |
| `splits/folds_{cvall,temporal}_seed0.json` | Fold assignment of real subjects, generated once (seed 0) and shared by all label variants. |
| `preprocess.slurm` | `nnUNetv2_plan_and_preprocess -pl nnUNetPlannerResEncM -c 3d_fullres` (CPU, `general`). |
| `train.slurm` | Array job, one fold per task: `nnUNetv2_train D 3d_fullres F -p nnUNetResEncUNetMPlans --npz`. Resumes with `--c` if a checkpoint exists. |
| `predict.slurm` | Exports out-of-fold probabilities; for datasets with `imagesTs`, 5-fold ensemble prediction with `--save_probabilities`. |
| `export_probs.py` | Turns nnU-Net `.npz` probabilities into NIfTI (`{sid}_marker_prob.nii.gz`, `{sid}_probs.nii.gz` with 3 classes, `{sid}_seg.nii.gz`) on the grid of the original image. |

## Designs and dataset IDs

* **cvall** (Design A): all subjects with a label in the variant (127 for `original`),
  5-fold CV. Every subject receives an out-of-fold prediction.
* **temporal** (Design B): the 102 train+val subjects of `splits_original.json` in
  5-fold CV; the 25 temporally later test subjects are `imagesTs` and are only
  predicted by the 5-fold ensemble.

Subjects missing from a variant are dropped from their fold; the remaining
fold assignment is unchanged. A synthetic case `{sid}_syn{k}` is only ever in the
training list of folds where `sid` is a training case (never validation; skipped
for temporal-test subjects).

ID = 500 + 10 * variant index + design code (cvall 1, temporal 2, cvall+synthetic 3,
temporal+synthetic 4). Variant indices are append-only (`VARIANT_INDEX` in `convert.py`).

| Variant | cvall | temporal | cvall+syn | temporal+syn |
|---|---|---|---|---|
| `original` (0) | 501 | 502 | 503 | 504 |
| `corrected` (1) | 511 | 512 | 513 | 514 |
| `original_sphere` (2) | 521 | 522 | 523 | 524 |

Names: `Dataset{ID}_SeedSeg{CvAll,Temporal}[Syn]_{variant}`.

## Running

```bash
source scripts/nnunet/env.sh
python scripts/nnunet/convert.py build --variant corrected --design cvall
sbatch --export=ALL,DATASET=511 scripts/nnunet/preprocess.slurm
sbatch --array=0-4 --export=ALL,DATASET=511 scripts/nnunet/train.slurm
# after all folds have finished:
sbatch --export=ALL,DATASET=511 scripts/nnunet/predict.slurm
```

For a quick test, add `TRAINER=nnUNetTrainer_5epochs` to `--export`.
Outputs go to `nnunet/predictions/{dataset}/{trainer}__{plans}/oof/` and `.../test_ensemble/`.

## Plans (nnUNetPlannerResEncM, 3d_fullres)

The same for every dataset built so far (501, 502, 511, 512, 521):

* Network: ResidualEncoderUNet (ResEnc M), 6 stages, features 32-64-128-256-320-320,
  first-stage kernel 1x3x3, first-stage stride 1x1x1, in-plane-only stride 1x2x2 in the last stage
* Target spacing: 2.0 x 0.6875 x 0.6875 mm (z, y, x), i.e. the native spacing
* Median image size after cropping: 40 x 317 x 316 voxels
* Patch size: 32 x 256 x 224 voxels; batch size 2
* Normalisation: per-image z-score; 3d_lowres not created (image too small)
* Training: default `nnUNetTrainer` (1000 epochs x 250 iterations, SGD with
  Nesterov momentum, initial lr 0.01 with poly decay, Dice + CE loss, deep supervision, default nnU-Net augmentation)
