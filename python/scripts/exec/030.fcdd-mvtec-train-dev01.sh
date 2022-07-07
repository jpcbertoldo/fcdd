#!/usr/bin/env bash
set -e  # stops the execution of a script if a command or pipeline has an 


cd $HOME

DEBUG_ARGS="--wandb-tags debug"
DEBUG_ARGS="${DEBUG_ARGS} --classes 0 --n-seeds 1"
# DEBUG_ARGS="${DEBUG_ARGS} --wandb-offline --wandb-checkpoint-mode none"
DEBUG_ARGS="${DEBUG_ARGS} --lightning-precision 16"

SBATCH_SCRIPT_FPATH="/cluster/CMM/home/jcasagrandebertoldo/repos/fcdd/python/scripts/sbatch/00-fcdd-mvtec-train-dev01-triple-run.sh"
echo ""
echo "SBATCH_SCRIPT_FPATH = ${SBATCH_SCRIPT_FPATH}"


# do old x new loss
# try 16bit


# launch

sbatch ${SBATCH_SCRIPT_FPATH} ${DEBUG_ARGS}
