#!/usr/bin/env bash
set -e  # stops the execution of a script if a command or pipeline has an 
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# ==============================================================================

SBATCH_SCRIPT_FPATH="${HOME}/repos/fcdd/python/scripts/sbatch/07-nparallel-arrayargs-01-1gpu-2cpuptask.slurm"

export SBATCH_SCRIPT_ARG_WORKDIR="${HOME}/fcdd/python/scripts/sweeps/004-fcdd-swa"
export SBATCH_SCRIPT_ARG_CONDAENV="fcdd_rc21"

export SBATCH_SCRIPT_ARG_SCRIPT_FNAME="null"
export SBATCH_SCRIPT_ARG_CUSTOM_COMMAND="wandb"

export SBATCH_SCRIPT_ARG_ARRAY_ARGS_FPATH="null"
export SBATCH_SCRIPT_ARG_NPARALLEL_RUNS=3

ARGS=""
ARGS="${ARGS} agent mines-paristech-cmm/fcdd-mvtec-bmvc-01/ji05j3me"
echo "ARGS=${ARGS}"

sbatch ${SBATCH_SCRIPT_FPATH} ${ARGS}
