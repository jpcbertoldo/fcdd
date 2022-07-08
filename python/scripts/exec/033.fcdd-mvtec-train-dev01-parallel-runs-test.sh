#!/usr/bin/env bash
set -e  # stops the execution of a script if a command or pipeline has an 


cd $HOME

DEBUG_ARGS=""
# DEBUG_ARGS="${DEBUG_ARGS} --n-seeds 1 --epochs 1"
# DEBUG_ARGS="${DEBUG_ARGS} --wandb-offline --wandb-checkpoint-mode none"
# DEBUG_ARGS="${DEBUG_ARGS} --wandb-tags debug"
DEBUG_ARGS="${DEBUG_ARGS} --wandb-tags deleteme"
echo "DEBUG_ARGS = ${DEBUG_ARGS}"

ARGS=""
ARGS="${ARGS} --wandb-project mvtec-debug" 
ARGS="${ARGS} --wandb-tags parallel-run-test"
ARGS="${ARGS} --n-seeds 1"
ARGS="${ARGS} --epochs 100"

SBATCH_SCRIPT_FPATH="/cluster/CMM/home/jcasagrandebertoldo/fcdd/python/scripts/sbatch/03-fcdd-mvtec-train-dev01-run-n-parallel.slurm"
echo "BATCH_SCRIPT_FPATH=${SBATCH_SCRIPT_FPATH}"

sbatch ${SBATCH_SCRIPT_FPATH} --slurm-n-parallel-runs 1 ${ARGS} ${DEBUG_ARGS} 
sleep 3