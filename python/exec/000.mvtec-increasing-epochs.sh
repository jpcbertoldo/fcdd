#!/usr/bin/env bash
set -e  # stops the execution of a script if a command or pipeline has an 

THIS_SCRIPT_DIR=$( cd ${0%/*} && pwd -P )
THIS_SCRIPT_DIR=$( realpath ${THIS_SCRIPT_DIR} )

WORKDIR=$(realpath ${THIS_SCRIPT_DIR}/../fcdd)
echo "WORKDIR=${WORKDIR}"

cd ${WORKDIR}
echo "pwd=$(pwd)"

# use my local conda
source ${HOME}/init-conda-bash
echo "which conda = $(which conda)"

conda activate fcdd_rc21
echo "\$CONDA_DEFAULT_ENV = $CONDA_DEFAULT_ENV"
echo "which python = $(which python)"

# mvtec default number of epochs is 200

GPU=0
MYTMPDIR="/data/bertoldo/tmp python"

SNAPSHOT_EPOCHS="0 1 2 4 9 19 49 99 199"
NITER=1

# 11 = toothbrush, the fastest to train
# CLASSES=( 11 )

# 0 = bottle
# 8 = pill
CLASSES=( 11 0 8 )

for CLASS_NUMBER in ${CLASSES[@]}
do 
    echo "CLASS_NUMBER=${CLASS_NUMBER}"
    CUDA_VISIBLE_DEVICES=${GPU} TMPDIR=${MYTMPDIR} python runners/run_mvtec.py --cls-restrictions ${CLASS_NUMBER} --snapshots-training ${SNAPSHOT_EPOCHS} --it ${NITER}
done