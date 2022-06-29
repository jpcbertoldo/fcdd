#!/usr/bin/env bash
set -e  # stops the execution of a script if a command or pipeline has an 

THIS_SCRIPT_DIR=$( cd ${0%/*} && pwd -P )
THIS_SCRIPT_DIR=$( realpath ${THIS_SCRIPT_DIR} )

cd $HOME
# debug params
# --cls-restrictions 0 --it 1 --epochs 1 --no-test

SBATCH_SCRIPT_FPATH="/cluster/CMM/home/jcasagrandebertoldo/repos/fcdd/python/scripts/sbatch/dev00/fcdd-mvtec-train-dev00-checkpoint02bis.sh"
echo ""
echo "SBATCH_SCRIPT_FPATH = ${SBATCH_SCRIPT_FPATH}"

SEMI_SUPERVISED_ARGS="--supervise-mode noise --noise-mode mvtec_gt --oe-limit 1"
echo ""
echo "\$SEMI_SUPERVISED_ARGS = ${SEMI_SUPERVISED_ARGS}"

# UNSUPERVISED_ARGS="--supervise-mode malformed_normal_gt --noise-mode confetti"
# echo ""
# echo "\$UNSUPERVISED_ARGS = ${UNSUPERVISED_ARGS}"




DEBUG_ARGS="--epochs 2 "
echo ""
echo "\$DEBUG_ARGS = ${DEBUG_ARGS}"


# launch

sbatch ${SBATCH_SCRIPT_FPATH} ${SEMI_SUPERVISED_ARGS} --loss-mode pixel-level-balanced-post-hubert ${DEBUG_ARGS}
