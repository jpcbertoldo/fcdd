#!/bin/bash
set -e  # stops the execution of a script if a command or pipeline has an 

THIS_SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
echo "THIS_SCRIPT_DIR=${THIS_SCRIPT_DIR}"

WORKDIR=${HOME}
echo "WORKDIR=${WORKDIR}"

cd ${WORKDIR}
echo "pwd=$(pwd)"

SBATCH_SCRIPT_FPATH="${THIS_SCRIPT_DIR}/../scripts/sbatch/sbatch-fcdd-mvtec.sh"
echo "SBATCH_SCRIPT_FPATH=${SBATCH_SCRIPT_FPATH}"

echo "sbatch=$(which sbatch)"
# mvtec default number of epochs is 200

MYTMPDIR="/mnt/data2/CMM/jpcasagrande/tmp"
echo "MYTMPDIR=$MYTMPDIR"

SNAPSHOT_EPOCHS="0 1 2 4 7 10 13 16 19 24 39 44 49 59 69 79 89 99 129 169 199"
NITER=1

NJOBS_PER_COMMAND=10

# TEST=1

# 11 = toothbrush, the fastest to train
# 0 = bottle
# 8 = pill
# 3 = carpet
# 10 = tile
# 6 = leather
# all the classes
CLASSES=( 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 )

echo "CLASSES=$CLASSES[@]"
echo "NITER=$NITER"
echo "SNAPSHOT_EPOCHS=$SNAPSHOT_EPOCHS"
echo "NJOBS_PER_COMMAND=$NJOBS_PER_COMMAND"


for CLASS_NUMBER in ${CLASSES[@]}
do 
    echo "CLASS_NUMBER=${CLASS_NUMBER}"

    ARGS="--cls-restrictions ${CLASS_NUMBER} --snapshots-training ${SNAPSHOT_EPOCHS} --it ${NITER}"
    echo "ARGS=${ARGS}"
    
    for JOB_LAUNCH_ITER in $( seq 1 ${NJOBS_PER_COMMAND[@]} )
    do
        echo "JOB_LAUNCH_ITER=$JOB_LAUNCH_ITER"

        if [[ $TEST ]]
        then
            echo "skipping"
        else
            TMPDIR=${MYTMPDIR} sbatch ${SBATCH_SCRIPT_FPATH} ${ARGS}
        fi
        
    done

done


if [[ $TEST ]]
then
    TMPDIR=${MYTMPDIR} sbatch ${SBATCH_SCRIPT_FPATH} --cls-restrictions 11 --it 1 --epochs 1
fi