#!/usr/bin/env bash
set -e  # stops the execution of a script if a command or pipeline has an 


cd $HOME

DEBUG_ARGS=""
DEBUG_ARGS="${DEBUG_ARGS} --n-seeds 1 --epochs 10"
DEBUG_ARGS="${DEBUG_ARGS} --wandb-offline --wandb-checkpoint-mode none"
DEBUG_ARGS="${DEBUG_ARGS} --wandb-tags debug"
echo "DEBUG_ARGS = ${DEBUG_ARGS}"

# "equivalent schedules"
# before, the epoch was manually defined as 10 cycles over the entire dataset
# now i just do 1, but i multiplied the number of epochs by 10
# it is still the same number of bathces
# !!!BUT the learning rate is different(lower) such that the learning rate at 
# the end is the same as it was in the previous versions
# before .985^50 = 47%
# now .985^500 = .05%
# now adapted .9985^500  = 47%
# ARGS="--wandb-project fcdd-mvtec-bmvc --wandb-tags bmvc-01"
# ARGS="${ARGS} --n-seeds 2"
# ARGS="${ARGS} --wandb-tags equivalent-schedules"
# ARGS="${ARGS} --scheduler-parameteres 0.9985"

# 4 runs per gpu
SBATCH_SCRIPT_FPATH="/cluster/CMM/home/jcasagrandebertoldo/fcdd/python/scripts/sbatch/02-fcdd-mvtec-train-dev01-4runspergpu-jobarray.slurm"

# triple run
# SBATCH_SCRIPT_FPATH="/cluster/CMM/home/jcasagrandebertoldo/fcdd/python/scripts/sbatch/01-fcdd-mvtec-train-dev01-triple-run-jobarray.slurm"

echo "SBATCH_SCRIPT_FPATH = ${SBATCH_SCRIPT_FPATH}"

# important
# EXPLAIN 4 runs per gpu
# EXPLAIN 4 runs per gpu
# EXPLAIN 4 runs per gpu
# EXPLAIN 4 runs per gpu
# EXPLAIN 4 runs per gpu

# do old x new loss
# try 16bit

# launch
sbatch ${SBATCH_SCRIPT_FPATH} ${ARGS} ${DEBUG_ARGS} --supervise-mode real-anomaly --loss old-fcdd
# sbatch ${SBATCH_SCRIPT_FPATH} ${ARGS} ${DEBUG_ARGS} --supervise-mode real-anomaly --loss pixelwise-batch-avg
# sbatch ${SBATCH_SCRIPT_FPATH} ${ARGS} ${DEBUG_ARGS} --supervise-mode synthetic-anomaly-confetti --loss old-fcdd
# sbatch ${SBATCH_SCRIPT_FPATH} ${ARGS} ${DEBUG_ARGS} --supervise-mode synthetic-anomaly-confetti --loss pixelwise-batch-avg

