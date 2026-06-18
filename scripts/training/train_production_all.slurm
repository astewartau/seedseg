#!/usr/bin/env bash
#SBATCH --partition=gpu_cuda
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=cuda80gb
#SBATCH --qos=gpu
#SBATCH --mem=100G
#SBATCH --time=15:45:00
#SBATCH --account=a_ai_collab
#SBATCH --output=logs/prod_all_%j.out
#SBATCH --error=logs/prod_all_%j.err

export PYTHONUNBUFFERED=1

echo "Starting production-all training on $(hostname) at $(date)"
echo "SEED=${SEED}"

# Test if the GPU is available
echo "Checking GPU availability"
echo `nvidia-smi`

if ! nvidia-smi | grep -q "NVIDIA-SMI"; then
    echo "No GPU available, exiting."
    exit 1
fi

# Set up conda
export CONDA_HOME=$HOME/miniconda3
source "$CONDA_HOME/etc/profile.d/conda.sh"
export LD_LIBRARY_PATH="$CONDA_HOME/envs/prostate/lib:$LD_LIBRARY_PATH"
conda activate prostate39

echo "Python: $(which python) ($(python --version))"
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

cd "$SLURM_SUBMIT_DIR"

python scripts/training/train_one_model.py T1 \
    --mode production \
    --data-dir data/all \
    --val-dir data/test/prepared \
    --val-subjects data/val_subjects.txt \
    --seed "$SEED" \
    --output-dir models/production-all
