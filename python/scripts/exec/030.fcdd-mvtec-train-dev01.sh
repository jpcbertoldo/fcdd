#!/usr/bin/env bash
set -e  # stops the execution of a script if a command or pipeline has an 


cd $HOME

DEBUG_ARGS=""
DEBUG_ARGS="${DEBUG_ARGS} --classes 0 --n-seeds 1 --epochs 1"
# DEBUG_ARGS="${DEBUG_ARGS} --wandb-offline --wandb-checkpoint-mode none"

SBATCH_SCRIPT_FPATH="/cluster/CMM/home/jcasagrandebertoldo/repos/fcdd/python/scripts/sbatch/00-fcdd-mvtec-train-dev01-triple-run.sh"
echo ""
echo "SBATCH_SCRIPT_FPATH = ${SBATCH_SCRIPT_FPATH}"


# SEMI_SUPERVISED_ARGS="--supervise-mode "
# echo ""
# echo "\$SEMI_SUPERVISED_ARGS = ${SEMI_SUPERVISED_ARGS}"

# UNSUPERVISED_ARGS="--supervise-mode "
# echo ""
# echo "\$UNSUPERVISED_ARGS = ${UNSUPERVISED_ARGS}"

# here
# here
# here
# here
# here
# here
# here
# here
# here
# here
# here
# do old x new loss
# try 16bit


# launch

sbatch ${SBATCH_SCRIPT_FPATH} ${DEBUG_ARGS}
