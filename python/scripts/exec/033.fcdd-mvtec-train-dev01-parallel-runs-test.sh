#!/usr/bin/env bash
set -e  # stops the execution of a script if a command or pipeline has an 


cd $HOME

DEBUG_ARGS=""
# DEBUG_ARGS="${DEBUG_ARGS} --n-seeds 1 --epochs 1"
# DEBUG_ARGS="${DEBUG_ARGS} --wandb-offline --wandb-checkpoint-mode none"
# DEBUG_ARGS="${DEBUG_ARGS} --wandb-tags debug"
# DEBUG_ARGS="${DEBUG_ARGS} --wandb-tags deleteme"
echo "DEBUG_ARGS = ${DEBUG_ARGS}"

ARGS=""
ARGS="${ARGS} --wandb-project mvtec-debug" 
ARGS="${ARGS} --wandb-tags parallel-run-test"
ARGS="${ARGS} --n-seeds 1"
ARGS="${ARGS} --epochs 100"
ARGS="${ARGS} --classes 0"

SBATCH_SCRIPT_FPATH="/cluster/CMM/home/jcasagrandebertoldo/fcdd/python/scripts/sbatch/03-fcdd-mvtec-train-dev01-run-n-parallel.slurm"
echo "BATCH_SCRIPT_FPATH=${SBATCH_SCRIPT_FPATH}"

# sbatch ${SBATCH_SCRIPT_FPATH} --slurm-n-parallel-runs 1 ${ARGS} ${DEBUG_ARGS} --wandb-tags nparallel:1 
# sleep 3

# sbatch ${SBATCH_SCRIPT_FPATH} --slurm-n-parallel-runs 2 ${ARGS} ${DEBUG_ARGS} --wandb-tags nparallel:2 
# sleep 3

# sbatch ${SBATCH_SCRIPT_FPATH} --slurm-n-parallel-runs 3 ${ARGS} ${DEBUG_ARGS} --wandb-tags nparallel:3 
# sleep 3

# using dft batch size (64 w/ accumulate 2) broke the runs (out of memory)
# i reduced them so i could fit 4 and 5 parallel proccesses

# nparallel:4
# 1/6 of the effective batch size (128/6=~21) 
sbatch ${SBATCH_SCRIPT_FPATH} --slurm-n-parallel-runs 4 ${ARGS} ${DEBUG_ARGS} --batch-size 21 --lightning-accumulate-grad-batches 6 --wandb-tags nparallel:4 
sleep 3

# nparallel:5
# 1/6 of the effective batch size (128/6=~21) 
# sbatch ${SBATCH_SCRIPT_FPATH} --slurm-n-parallel-runs 5 ${ARGS} ${DEBUG_ARGS} --batch-size 21 --lightning-accumulate-grad-batches 6 --wandb-tags nparallel:5 
# sleep 3
