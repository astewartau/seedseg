# Source from Slurm scripts: activates the env and sets nnU-Net paths.
export CONDA_HOME=$HOME/miniconda3
source "$CONDA_HOME/etc/profile.d/conda.sh"
conda activate seedseg-nnunet
export NNUNET_BASE=/scratch/user/uqaste15/prostate-v2/nnunet
export nnUNet_raw=$NNUNET_BASE/nnUNet_raw
export nnUNet_preprocessed=$NNUNET_BASE/nnUNet_preprocessed
export nnUNet_results=$NNUNET_BASE/nnUNet_results
export PYTHONUNBUFFERED=1
export SEEDSEG_REPO=/scratch/user/uqaste15/src/seedseg-nnunet
