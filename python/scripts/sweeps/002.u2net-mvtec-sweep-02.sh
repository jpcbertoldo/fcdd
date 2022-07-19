#!/usr/bin/env bash
set -e  # stops the execution of a script if a command or pipeline has an 

cd $HOME

ARGS=""
ARGS="${ARGS} mines-paristech-cmm/unetdd-mvtec-00/73omzk4h"

echo "ARGS=${ARGS}"

SBATCH_SCRIPT_FPATH="/cluster/CMM/home/jcasagrandebertoldo/fcdd/python/scripts/sbatch/04-sweep-agent-n-parallel.slurm"
echo "BATCH_SCRIPT_FPATH=${SBATCH_SCRIPT_FPATH}"

sbatch ${SBATCH_SCRIPT_FPATH} --slurm-n-parallel-runs 3 ${ARGS} 
