#!/usr/bin/env bash
set -e  # stops the execution of a script if a command or pipeline has an 
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# ==============================================================================

SBATCH_SCRIPT_FPATH="${HOME}/repos/fcdd/python/scripts/sbatch/08-nparallel-arrayargs-01-1gpu-4cpuptask.slurm"

export SBATCH_SCRIPT_ARG_ARRAY_ARGS_FPATH="${HOME}/repos/fcdd/python/scripts/sweeps/004-fcdd-swa/sweep-003-then-004.agent-slurm.args"
export SBATCH_SCRIPT_ARG_NPARALLEL_RUNS=3

export SBATCH_SCRIPT_ARG_WORKDIR="${HOME}/fcdd/python/dev"
export SBATCH_SCRIPT_ARG_CONDAENV="fcdd_rc21"

export SBATCH_SCRIPT_ARG_COMMAND="wandb"

ARGS=""
# ARGS="${ARGS}"
echo "ARGS=${ARGS}"

sbatch ${SBATCH_SCRIPT_FPATH} ${ARGS}
