#!/usr/bin/env bash
set -e  # stops the execution of a script if a command or pipeline has an 


cd $HOME

DEBUG_ARGS=""
# DEBUG_ARGS="${DEBUG_ARGS} --n-seeds 1 --epochs 1"
# DEBUG_ARGS="${DEBUG_ARGS} --wandb-offline --wandb-checkpoint-mode none"
# DEBUG_ARGS="${DEBUG_ARGS} --wandb-tags debug"
echo "DEBUG_ARGS = ${DEBUG_ARGS}"

# "equivalent schedules"
# before, the epoch was manually defined as 10 cycles over the entire dataset
# now i just do 1, but i multiplied the number of epochs by 10
# it is still the same number of bathces
# !!!BUT the learning rate is different (lower) such that the learning rate at 
# the end is the same as it was in the previous versions
# before .985^50 = 47%
# now .985^500 = .05%
# now adapted .9985^500  = 47%
ARGS="--wandb-project fcdd-mvtec-bmvc --wandb-tags bmvc-01"
# ARGS="${ARGS} --n-seeds 5"

# with old rate
# ARGS="${ARGS} --wandb-tags old-lr-schedule-rate"
# ARGS="${ARGS} --scheduler-parameters 0.985"

# with equivalent rate
ARGS="${ARGS} --wandb-tags equivalent-lr-schedule-rate"
ARGS="${ARGS} --scheduler-parameters 0.9985"

# n-parallel runs
SBATCH_SCRIPT_FPATH="/cluster/CMM/home/jcasagrandebertoldo/repos/fcdd/python/scripts/sbatch/05-fcdd-mvtec-train-dev01-run-n-parallel.slurm"

echo "SBATCH_SCRIPT_FPATH = ${SBATCH_SCRIPT_FPATH}"

# these two configs had some runs missing 
# these two configs had some runs missing 
# these two configs had some runs missing 
# these two configs had some runs missing 

# launch
sbatch ${SBATCH_SCRIPT_FPATH} ${ARGS} ${DEBUG_ARGS} --supervise-mode real-anomaly --loss old-fcdd --classes 4 --n-seeds 4
sleep 5

sbatch ${SBATCH_SCRIPT_FPATH} ${ARGS} ${DEBUG_ARGS} --supervise-mode synthetic-anomaly-confetti --loss old-fcdd --classes 6 --n-seeds 3
sleep 5
