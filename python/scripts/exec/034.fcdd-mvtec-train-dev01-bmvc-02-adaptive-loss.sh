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
ARGS="${ARGS} --n-seeds 1"

# with old rate
# ARGS="${ARGS} --wandb-tags old-lr-schedule-rate"
# ARGS="${ARGS} --scheduler-parameters 0.985"

# with equivalent rate
ARGS="${ARGS} --wandb-tags equivalent-lr-schedule-rate"
ARGS="${ARGS} --scheduler-parameters 0.9985"

ARGS="${ARGS} --loss pixelwise-batch-avg-clip-score-cdf-adaptive"
ARGS="${ARGS} --wandb-tags adaptive-loss"

# n-parallel runs
# SBATCH_SCRIPT_FPATH="/cluster/CMM/home/jcasagrandebertoldo/repos/fcdd/python/scripts/sbatch/05-fcdd-mvtec-train-dev01-run-n-parallel.slurm"

# this sets env variables for the adaptive clipping loss
SBATCH_SCRIPT_FPATH="/cluster/CMM/home/jcasagrandebertoldo/repos/fcdd/python/scripts/sbatch/05-fcdd-mvtec-train-dev01-run-n-parallel.adaptive-loss-envvar.slurm"


echo "BATCH_SCRIPT_FPATH=${SBATCH_SCRIPT_FPATH}"

ARGS_PARALLEL_RUNS_01="--slurm-n-parallel-runs 1 --wandb-tags nparallel-runs:1"
ARGS_PARALLEL_RUNS_02="--slurm-n-parallel-runs 2 --wandb-tags nparallel-runs:2"
ARGS_PARALLEL_RUNS_03="--slurm-n-parallel-runs 3 --wandb-tags nparallel-runs:3"
ARGS_PARALLEL_RUNS_04="--slurm-n-parallel-runs 4 --wandb-tags nparallel-runs:4"
ARGS_PARALLEL_RUNS_05="--slurm-n-parallel-runs 5 --wandb-tags nparallel-runs:5"
ARGS_PARALLEL_RUNS_06="--slurm-n-parallel-runs 6 --wandb-tags nparallel-runs:6"

# seq args: start step end
for i in $(seq 0 1 2)
do
    if [[ $i -eq 0 ]]; then
        # PARALLEL_RUNS=${ARGS_PARALLEL_RUNS_01}
        PARALLEL_RUNS=${ARGS_PARALLEL_RUNS_02}
        # PARALLEL_RUNS=${ARGS_PARALLEL_RUNS_03}
    elif [[ $i -eq 1 ]]; then
        PARALLEL_RUNS=${ARGS_PARALLEL_RUNS_02}
    else
        PARALLEL_RUNS=${ARGS_PARALLEL_RUNS_01}
        # PARALLEL_RUNS=${ARGS_PARALLEL_RUNS_02}
        # exit
    fi

    echo "i=${i}"
    echo "PARALLEL_RUNS=${PARALLEL_RUNS}"
    
    sbatch ${SBATCH_SCRIPT_FPATH} ${PARALLEL_RUNS} ${ARGS} ${DEBUG_ARGS} --supervise-mode real-anomaly
    sbatch ${SBATCH_SCRIPT_FPATH} ${PARALLEL_RUNS} ${ARGS} ${DEBUG_ARGS} --supervise-mode synthetic-anomaly-confetti

done
