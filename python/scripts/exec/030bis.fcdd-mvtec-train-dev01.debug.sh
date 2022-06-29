#!/usr/bin/env bash
set -e  # stops the execution of a script if a command or pipeline has an 


cd $HOME
# debug params
# --cls-restrictions 0 --it 1 --epochs 1 --no-test

SBATCH_SCRIPT_FPATH="/cluster/CMM/home/jcasagrandebertoldo/fcdd/python/scripts/sbatch/00bis-fcdd-mvtec-train-dev01-triple-run-DEBUG.sh"
echo ""
echo "SBATCH_SCRIPT_FPATH = ${SBATCH_SCRIPT_FPATH}"


# launch

sbatch ${SBATCH_SCRIPT_FPATH}
sleep 7  # avoid the same directory name
