#!/usr/bin/env bash
set -e  # stops the execution of a script if a command or pipeline has an 


cd $HOME

ARGS=""

# old
# ARGS="${ARGS} mines-paristech-cmm/unetdd-mvtec-00/8g7n2cc5"

ARGS="${ARGS} mines-paristech-cmm/unetdd-mvtec-00/7e6qg2bj"

echo "ARGS=${ARGS}"

SBATCH_SCRIPT_FPATH="/cluster/CMM/home/jcasagrandebertoldo/fcdd/python/scripts/sbatch/04-sweep-agent-n-parallel.slurm"
echo "BATCH_SCRIPT_FPATH=${SBATCH_SCRIPT_FPATH}"

sbatch ${SBATCH_SCRIPT_FPATH} --slurm-n-parallel-runs 2 ${ARGS} 
sbatch ${SBATCH_SCRIPT_FPATH} --slurm-n-parallel-runs 3 ${ARGS} 
sbatch ${SBATCH_SCRIPT_FPATH} --slurm-n-parallel-runs 4 ${ARGS} 
