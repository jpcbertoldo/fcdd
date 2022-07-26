#!/usr/bin/env bash
set -e  # stops the execution of a script if a command or pipeline has an 
cd $HOME


SBATCH_SCRIPT_FPATH="/cluster/CMM/home/jcasagrandebertoldo/fcdd/python/scripts/sbatch/06-nparallel-arrayargs-00-1gpu-2cpuptask-1jobsimult.slurm"

export SBATCH_SCRIPT_ARG_WORKDIR="${HOME}/fcdd/python/scripts/sbatch/debug"
export SBATCH_SCRIPT_ARG_SCRIPT_FNAME="printargs.py"
export SBATCH_SCRIPT_ARG_CONDAENV="fcdd_rc21"

export SBATCH_SCRIPT_ARG_NPARALLEL_RUNS=1
# export SBATCH_SCRIPT_ARG_ARRAY_ARGS_FPATH="${HOME}/fcdd/python/scripts/sbatch/test/06.arrayargs.txt"
export SBATCH_SCRIPT_ARG_ARRAY_ARGS_FPATH="${HOME}/fcdd/python/scripts/sbatch/test/06.arrayargs-empty.txt"

ARGS="--arg1-from-exec 1 --arg2-from-exec 2"

sbatch ${SBATCH_SCRIPT_FPATH} ${ARGS} 

# invalid array args fpath