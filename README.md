# Prostate gold seed segmentation

Detection of gold fiducial markers in T1-weighted prostate MRI. The deployed model is an nnU-Net v2 (residual encoder, 3d_fullres) 5-fold ensemble exported to ONNX; the web app and the OpenRecon container run the same pipeline (`scripts/deploy/seedseg_onnx.py`).

## Structure

- `scripts/` - Python pipeline: preprocessing, training, inference, evaluation, analysis
- `web/` - Browser-based inference app (ONNX Runtime Web + NiiVue)
- `models/` - PyTorch model checkpoints (not tracked)
- `paper_draft/` - Manuscript and figures

## Web app

The `web/` directory contains a browser-based version of the inference pipeline. All computation runs locally in the browser using ONNX Runtime Web (WASM backend). No data is uploaded to any server.

### Setup

```bash
# 1. Download ONNX Runtime WASM files
cd web && bash setup.sh

# 2. Put the ONNX folds and seedseg-v2.json in web/models/, e.g. export them from a
#    trained nnU-Net model folder (needs nnunetv2, onnx, onnxconverter-common):
python ../scripts/deploy/export_onnx.py --model <nnUNet_results/Dataset.../nnUNetTrainer__nnUNetResEncUNetMPlans__3d_fullres> \
    --out models
python ../scripts/deploy/make_w16.py models   # fp16-stored weights (the web app's default download)

# 3. Start development server
cd web && bash run.sh
```

Then open http://localhost:8080 and upload a T1-weighted prostate MRI (NIfTI or DICOM).

## Python pipeline

### Training

```bash
python scripts/training/train_one_model.py T1 --mode production --seed 42 \
    --data-dir data/train --val-dir data/test/prepared \
    --val-subjects data/val_subjects.txt --output-dir models/
```

### Inference

```bash
# ONNX Runtime (CPU or --device cuda); input: T1 NIfTI without bias correction
python scripts/deploy/seedseg_onnx.py --models <dir with seedseg-v2.json> --input t1.nii --out results/sub \
    [--k 3] [--folds 0 1 2 3 4] [--no-mirror]
```

Outputs: `results/sub_marker_prob.nii.gz` and `results/sub_detections.json` (top-K markers with mean probability, centroid in voxel and world coordinates, and `count_flag` when fewer than K candidates were found).

Tests (wiring only, tiny data): `pytest scripts/deploy/tests` and `node --test web/tests/*.test.mjs`.

## Citation

Stewart et al. "Deep-Learning-Enabled Differentiation between Intraprostatic Gold Fiducial Markers and Calcification in Quantitative Susceptibility Mapping." bioRxiv (2023). https://doi.org/10.1101/2023.10.26.564293
