#!/usr/bin/env bash
set -e  # stops the execution of a script if a command or pipeline has an 

THIS_SCRIPT_DIR=$( cd ${0%/*} && pwd -P )
THIS_SCRIPT_DIR=$( realpath ${THIS_SCRIPT_DIR} )

cd $HOME
# debug params
# --cls-restrictions 0 --it 1 --epochs 1 --no-test

# // B) unsupervised (confetti), MASK, (FIXED) pixel level loss
SBATCH_SCRIPT_FPATH="/cluster/CMM/home/jcasagrandebertoldo/repos/fcdd/python/scripts/sbatch/fcdd-mvtec-train-dev00-checkpoint01.relaunch02.sh"
sbatch ${SBATCH_SCRIPT_FPATH}
echo "done"



