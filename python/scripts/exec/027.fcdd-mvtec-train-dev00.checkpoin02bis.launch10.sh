#!/usr/bin/env bash
set -e  # stops the execution of a script if a command or pipeline has an 

THIS_SCRIPT_DIR=$( cd ${0%/*} && pwd -P )
THIS_SCRIPT_DIR=$( realpath ${THIS_SCRIPT_DIR} )

cd $HOME
# debug params
# --cls-restrictions 0 --it 1 --epochs 1 --no-test

SBATCH_SCRIPT_FPATH="/cluster/CMM/home/jcasagrandebertoldo/repos/fcdd/python/scripts/sbatch/fcdd-mvtec-train-dev00-checkpoint02bis.sh"
echo ""
echo "SBATCH_SCRIPT_FPATH = ${SBATCH_SCRIPT_FPATH}"

SEMI_SUPERVISED_ARGS="--supervise-mode noise --noise-mode mvtec_gt --oe-limit 1"
echo ""
echo "\$SEMI_SUPERVISED_ARGS = ${SEMI_SUPERVISED_ARGS}"

UNSUPERVISED_ARGS="--supervise-mode malformed_normal_gt --noise-mode confetti"
echo ""
echo "\$UNSUPERVISED_ARGS = ${UNSUPERVISED_ARGS}"


# launch

sbatch ${SBATCH_SCRIPT_FPATH} ${UNSUPERVISED_ARGS} --loss-mode 
sleep 15  # avoid the same directory name

sbatch ${SBATCH_SCRIPT_FPATH} ${SEMI_SUPERVISED_ARGS} --loss-mode p
sleep 15  # avoid the same directory name


# --loss-mode 
# LOSS_PIXEL_LEVEL_BALANCED = "pixel-level-balanced"
# LOSS_PIXEL_WISE_AVERAGE_DISTANCE_PER_IMAGE = "pixel-wise-average-distance-per-image"
# LOSS_PIXEL_WISE_AVERAGES_PER_IMAGE = "pixel-wise-averages-per-image"
# LOSS_PIXEL_WISE_AVERAGES_PER_IMAGE_BALANCED = "pixel-wise-averages-per-image-balanced"

# NOT GONNA LAUNCH
# LOSS_PIXEL_LEVEL_FOCAL = "pixel-level-focal"
# LOSS_PIXEL_LEVEL_FOCAL2 = "pixel-level-focal2"
# LOSS_PIXEL_LEVEL = "pixel-level"
# LOSS_PIXEL_WISE_AVERAGE_DISTANCES = "pixel-wise-average-distances"


# --net FCDD_CNN224_VGG_512_F