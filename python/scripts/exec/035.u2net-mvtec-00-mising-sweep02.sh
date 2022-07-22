#!/usr/bin/env bash
set -e  # stops the execution of a script if a command or pipeline has an 
cd $HOME
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

SBATCH_SCRIPT_FPATH="/cluster/CMM/home/jcasagrandebertoldo/fcdd/python/scripts/sbatch/06-nparallel-arrayargs-01-1gpu-2cpuptask-3jobsimult.slurm"

export SBATCH_SCRIPT_ARG_WORKDIR="${HOME}/fcdd/python/dev"
export SBATCH_SCRIPT_ARG_CONDAENV="fcdd_rc21"

# check me
# check me
# check me
# check me
# check me
# check me
export SBATCH_SCRIPT_ARG_SCRIPT_FNAME="train_mvtec_dev01.py"

export SBATCH_SCRIPT_ARG_ARRAY_ARGS_FPATH="${SCRIPT_DIR}/035.u2net-mvtec-00-arrayargs"
export SBATCH_SCRIPT_ARG_NPARALLEL_RUNS=3

ARGS="--wandb-project fcdd-mvtec-bmvc --wandb-tags bmvc-01"
ARGS="${ARGS} --n-seeds 1"

# COMMON_ARGS=""
# COMMON_ARGS="${COMMON_ARGS} --nworkers 1"
# COMMON_ARGS="${COMMON_ARGS} --wandb-tags dev01 run-n-parallel-${NPARALLEL_RUNS}"
# echo "COMMON_ARGS=${COMMON_ARGS}"


DEBUG_ARGS=""
# DEBUG_ARGS="${DEBUG_ARGS} --n-seeds 1 --epochs 1"
# DEBUG_ARGS="${DEBUG_ARGS} --wandb-offline --wandb-checkpoint-mode none"
# DEBUG_ARGS="${DEBUG_ARGS} --wandb-tags debug"
# DEBUG_ARGS="${DEBUG_ARGS} --wandb-tags deleteme"
echo "DEBUG_ARGS=${DEBUG_ARGS}"

sbatch ${SBATCH_SCRIPT_FPATH} ${ARGS} ${DEBUG_ARGS}
