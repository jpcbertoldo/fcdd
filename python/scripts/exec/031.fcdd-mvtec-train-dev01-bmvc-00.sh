#!/usr/bin/env bash
set -e  # stops the execution of a script if a command or pipeline has an 


cd $HOME

DEBUG_ARGS=""
# DEBUG_ARGS="${DEBUG_ARGS} --n-seeds 1 --epochs 1"
# DEBUG_ARGS="${DEBUG_ARGS} --wandb-offline --wandb-checkpoint-mode none"
# DEBUG_ARGS="${DEBUG_ARGS} --wandb-tags debug"
echo "DEBUG_ARGS = ${DEBUG_ARGS}"

ARGS="--wandb-project fcdd-mvtec-bmvc --wandb-tags bmvc-00"
ARGS="${ARGS} --n-seeds 2"

SBATCH_SCRIPT_FPATH="/cluster/CMM/home/jcasagrandebertoldo/fcdd/python/scripts/sbatch/01-fcdd-mvtec-train-dev01-triple-run-jobarray.slurm"
echo "SBATCH_SCRIPT_FPATH = ${SBATCH_SCRIPT_FPATH}"

# important
# each sbatch script called here will launch 2 jobs at a time and enqueue 
# the 15 classes with 2 supervision modes = 30 jobs in total
# and each job will launch 3 trainings 

# do old x new loss
# try 16bit

# launch
sbatch ${SBATCH_SCRIPT_FPATH} ${ARGS} ${DEBUG_ARGS} --loss old-fcdd
sbatch ${SBATCH_SCRIPT_FPATH} ${ARGS} ${DEBUG_ARGS} --loss pixelwise-batch-avg