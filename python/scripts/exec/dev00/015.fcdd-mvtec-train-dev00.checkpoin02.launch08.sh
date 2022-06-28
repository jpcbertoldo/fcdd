#!/usr/bin/env bash
set -e  # stops the execution of a script if a command or pipeline has an 

THIS_SCRIPT_DIR=$( cd ${0%/*} && pwd -P )
THIS_SCRIPT_DIR=$( realpath ${THIS_SCRIPT_DIR} )

cd $HOME
# debug params
# --cls-restrictions 0 --it 1 --epochs 1 --no-test

SBATCH_SCRIPT_FPATH="/cluster/CMM/home/jcasagrandebertoldo/repos/fcdd/python/scripts/sbatch/fcdd-mvtec-train-dev00-checkpoint02.sh"


# supervised, mask, pixel level loss
sbatch ${SBATCH_SCRIPT_FPATH} --supervise-mode malformed_normal_gt --noise-mode confetti --loss-mode pixel-wise-average-distances
sleep 3  # avoid the same directory name

# --loss-mode 

# LOSS_PIXEL_LEVEL = "pixel-level"
# LOSS_PIXEL_LEVEL_BALANCED = "pixel-level-balanced"
# LOSS_PIXEL_LEVEL_FOCAL = "pixel-level-focal"
# LOSS_PIXEL_WISE_AVERAGE_DISTANCES = "pixel-wise-average-distances"
