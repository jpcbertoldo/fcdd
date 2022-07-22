#!/usr/bin/env bash
set -e  # stops the execution of a script if a command or pipeline has an 
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# ==============================================================================

SBATCH_SCRIPT_FPATH="${HOME}/fcdd/python/scripts/sbatch/07-nparallel-arrayargs-02-1gpu-4cpuptask.slurm"

export SBATCH_SCRIPT_ARG_WORKDIR="${HOME}/fcdd/python/scripts"
export SBATCH_SCRIPT_ARG_CONDAENV="fcdd_rc21"

export SBATCH_SCRIPT_ARG_SCRIPT_FNAME="null"
export SBATCH_SCRIPT_ARG_CUSTOM_COMMAND="wandb"

export SBATCH_SCRIPT_ARG_ARRAY_ARGS_FPATH="null"
export SBATCH_SCRIPT_ARG_NPARALLEL_RUNS=3

ARGS=""
ARGS="${ARGS} agent mines-paristech-cmm/fcdd-mvtec-bmvc-01/0cmmty3c"
echo "ARGS=${ARGS}"

sbatch ${SBATCH_SCRIPT_FPATH} ${ARGS}
