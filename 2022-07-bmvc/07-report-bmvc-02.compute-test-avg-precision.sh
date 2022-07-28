#!/usr/bin/env bash
set -e  # stops the execution of a script if a command or pipeline has an 
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# ==============================================================================

SBATCH_SCRIPT_FPATH="${HOME}/repos/fcdd/python/scripts/sbatch/08-nparallel-arrayargs-02-1gpu-4cpuptask.slurm"

export SBATCH_SCRIPT_ARG_ARRAY_ARGS_FPATH="null"
export SBATCH_SCRIPT_ARG_NPARALLEL_RUNS=1

export SBATCH_SCRIPT_ARG_WORKDIR="${HOME}/fcdd/2022-07-bmvc"
export SBATCH_SCRIPT_ARG_CONDAENV="fcdd_rc21"

export SBATCH_SCRIPT_ARG_COMMAND="python"

ARGS=""
ARGS="${ARGS} 07-report-bmvc-02.compute-test-avg-precision.py"
echo "ARGS=${ARGS}"

sbatch ${SBATCH_SCRIPT_FPATH} ${ARGS}
