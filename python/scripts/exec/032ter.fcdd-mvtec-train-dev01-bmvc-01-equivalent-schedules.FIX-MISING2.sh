#!/usr/bin/env bash
set -e  # stops the execution of a script if a command or pipeline has an 


cd $HOME

DEBUG_ARGS=""
# DEBUG_ARGS="${DEBUG_ARGS} --n-seeds 1 --epochs 1"
# DEBUG_ARGS="${DEBUG_ARGS} --wandb-offline --wandb-checkpoint-mode none"
# DEBUG_ARGS="${DEBUG_ARGS} --wandb-tags debug"
echo "DEBUG_ARGS = ${DEBUG_ARGS}"

ARGS=""
# "equivalent schedules"
# before, the epoch was manually defined as 10 cycles over the entire dataset
# now i just do 1, but i multiplied the number of epochs by 10
# it is still the same number of bathces
# !!!BUT the learning rate is different (lower) such that the learning rate at 
# the end is the same as it was in the previous versions
# before .985^50 = 47%
# now .985^500 = .05%
# now adapted .9985^500  = 47%

ARGS="${ARGS} --wandb-project fcdd-mvtec-bmvc"
ARGS="${ARGS} --wandb-tags bmvc-01"
# ARGS="${ARGS} --n-seeds 5"

# with old rate
OLD_RATE_ARGS="${ARGS} --wandb-tags old-lr-schedule-rate"
OLD_RATE_ARGS="${ARGS} --scheduler-parameters 0.985"

# with equivalent rate
EQUIVALENT_RATE_ARGS="${ARGS} --wandb-tags equivalent-lr-schedule-rate"
EQUIVALENT_RATE_ARGS="${ARGS} --scheduler-parameters 0.9985"

ARGS_PARALLEL_RUNS_01="--slurm-n-parallel-runs 1 --wandb-tags nparallel-runs:1"
ARGS_PARALLEL_RUNS_02="--slurm-n-parallel-runs 2 --wandb-tags nparallel-runs:2"
ARGS_PARALLEL_RUNS_03="--slurm-n-parallel-runs 3 --wandb-tags nparallel-runs:3"
ARGS_PARALLEL_RUNS_04="--slurm-n-parallel-runs 4 --wandb-tags nparallel-runs:4"
ARGS_PARALLEL_RUNS_05="--slurm-n-parallel-runs 5 --wandb-tags nparallel-runs:5"
ARGS_PARALLEL_RUNS_06="--slurm-n-parallel-runs 6 --wandb-tags nparallel-runs:6"


# n-parallel runs
SBATCH_SCRIPT_FPATH="/cluster/CMM/home/jcasagrandebertoldo/repos/fcdd/python/scripts/sbatch/05-fcdd-mvtec-train-dev01-run-n-parallel.slurm"

echo "SBATCH_SCRIPT_FPATH = ${SBATCH_SCRIPT_FPATH}"

# launch



################################################################################
# [2022-07-20] 12h
# 12 confetti old-fcdd 0.985(old rate) ==> 5 ==> o launch
# 13 confetti old-fcdd 0.985(old rate) ==> 5 ==> o launch
# 14 confetti old-fcdd 0.985(old rate) ==> 5 ==> o launch


# 14 confetti pixelwise 0.9985(equiv rate) ==> 4 OK


sbatch ${SBATCH_SCRIPT_FPATH} ${ARGS_PARALLEL_RUNS_02} ${ARGS} ${OLD_RATE_ARGS} ${DEBUG_ARGS} --supervise-mode synthetic-anomaly-confetti --loss old-fcdd --classes 12 13 14 --n-seeds 1

sbatch ${SBATCH_SCRIPT_FPATH} ${ARGS_PARALLEL_RUNS_02} ${ARGS} ${OLD_RATE_ARGS} ${DEBUG_ARGS} --supervise-mode synthetic-anomaly-confetti --loss old-fcdd --classes 12 13 14 --n-seeds 1

sbatch ${SBATCH_SCRIPT_FPATH} ${ARGS_PARALLEL_RUNS_01} ${ARGS} ${OLD_RATE_ARGS} ${DEBUG_ARGS} --supervise-mode synthetic-anomaly-confetti --loss old-fcdd --classes 12 13 14 --n-seeds 1