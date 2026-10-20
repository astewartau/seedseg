# Source from Slurm scripts: activates the env and sets nnU-Net paths.
export CONDA_HOME=$HOME/miniconda3
_u=$-; set +u  # conda activate is not nounset-safe
source "$CONDA_HOME/etc/profile.d/conda.sh"
conda activate seedseg-nnunet
[[ $_u == *u* ]] && set -u
export NNUNET_BASE=/scratch/user/uqaste15/prostate-v2/nnunet
export nnUNet_raw=$NNUNET_BASE/nnUNet_raw
export nnUNet_preprocessed=$NNUNET_BASE/nnUNet_preprocessed
# RESULTS_TAG=<tag> keeps repeat runs of the same trainer apart (nnUNet_results_<tag>, predictions_<tag>)
export nnUNet_results=$NNUNET_BASE/nnUNet_results${RESULTS_TAG:+_$RESULTS_TAG}
export PYTHONUNBUFFERED=1
export SEEDSEG_REPO=/scratch/user/uqaste15/src/seedseg-nnunet
