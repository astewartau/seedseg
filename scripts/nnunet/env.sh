# Source from Slurm scripts: puts the seedseg-train env first on PATH and sets nnU-Net paths.
# The env is a uv venv built from envs/seedseg-train.lock.txt (see envs/README.md).
SCR=${SCR:-/scratch/user/uqaste15}
export PATH=$SCR/projects/prostate/env/seedseg-train/bin:$PATH
export NNUNET_BASE=$SCR/projects/prostate/v2/nnunet
export nnUNet_raw=$NNUNET_BASE/nnUNet_raw
export nnUNet_preprocessed=$NNUNET_BASE/nnUNet_preprocessed
# RESULTS_TAG=<tag> keeps repeat runs of the same trainer apart (nnUNet_results_<tag>, predictions_<tag>)
export nnUNet_results=$NNUNET_BASE/nnUNet_results${RESULTS_TAG:+_$RESULTS_TAG}
export PYTHONUNBUFFERED=1
# The clone of this repo on Bunya (override for a worktree: SEEDSEG_REPO=~/code/.worktrees/seedseg-<branch>)
export SEEDSEG_REPO=${SEEDSEG_REPO:-$HOME/code/seedseg}
