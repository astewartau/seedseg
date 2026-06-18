#!/usr/bin/env bash
#SBATCH --partition=gpu_cuda
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --account=a_ai_collab
#SBATCH --output=logs/time_inference_%j.out
#SBATCH --error=logs/time_inference_%j.err

export PYTHONUNBUFFERED=1

echo "=== INFERENCE TIMING TEST ==="
echo "Host: $(hostname)"
echo "Date: $(date)"
nvidia-smi

export CONDA_HOME=$HOME/miniconda3
source "$CONDA_HOME/etc/profile.d/conda.sh"
conda activate prostate39

cd /scratch/user/uqaste15/2024-prostate

MODELS=(models/production/T1-*-best.pth)
INPUT=data/test/prepared/z0008611/z0008611_MRI_homogeneity-corrected.nii
OUTDIR=results/timing_test

echo ""
echo "Models: ${MODELS[@]}"
echo "Input: $INPUT"
echo ""

# Time GPU inference (4 models, consensus)
echo "=== GPU INFERENCE (4 models) ==="
time python scripts/inference/consensus_inference.py \
  --models "${MODELS[@]}" \
  --input "$INPUT" \
  --output "$OUTDIR/gpu_4model" \
  --n-markers 3

# Time single model inference
echo ""
echo "=== GPU INFERENCE (1 model) ==="
time python scripts/inference/consensus_inference.py \
  --models "${MODELS[0]}" \
  --input "$INPUT" \
  --output "$OUTDIR/gpu_1model" \
  --n-markers 3

# Time CPU inference (4 models) - set CUDA_VISIBLE_DEVICES to force CPU
echo ""
echo "=== CPU INFERENCE (4 models) ==="
CUDA_VISIBLE_DEVICES="" time python scripts/inference/consensus_inference.py \
  --models "${MODELS[@]}" \
  --input "$INPUT" \
  --output "$OUTDIR/cpu_4model" \
  --n-markers 3

echo ""
echo "=== DONE ==="
echo "Finished at $(date)"
