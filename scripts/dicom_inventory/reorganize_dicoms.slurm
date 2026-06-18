#!/bin/bash
#SBATCH --job-name=reorg-dicoms
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --account=a_ai_collab
#SBATCH --output=/home/uqaste15/data/2024-prostate/logs/reorganize_%j.out

export PYTHONUNBUFFERED=1

# Set up conda
export CONDA_HOME=$HOME/miniconda3
source "$CONDA_HOME/etc/profile.d/conda.sh"
conda activate prostate39

echo "Python: $(which python)"
echo "Starting reorganization at $(date)"

cd /home/uqaste15/data/2024-prostate
python scripts/dicom_inventory/reorganize_dicoms.py

echo "Finished at $(date)"
