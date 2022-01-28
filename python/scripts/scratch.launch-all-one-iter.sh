
#!/usr/bin/env bash
set -e  # stops the execution of a script if a command or pipeline has an 

THIS_SCRIPT_DIR=$( cd ${0%/*} && pwd -P )
THIS_SCRIPT_DIR=$( realpath ${THIS_SCRIPT_DIR} )

WORKDIR=$(realpath ${THIS_SCRIPT_DIR}/../fcdd)
cd ${WORKDIR}
echo "pwd=$(pwd)"

# use my local conda
source ${HOME}/init-conda-bash
echo "which conda = $(which conda)"

conda activate fcdd_rc21
echo "\$CONDA_DEFAULT_ENV = $CONDA_DEFAULT_ENV"
echo "which python = $(which python)"

###############################################################
FILE="fmnist-cifar100.log"
CLASSES_LIST=( $( seq 0 9 ) )

echo "launching fmnist-cifar100"
touch ${FILE}

for CLASS in "${CLASSES_LIST[@]}"
do
    echo "CLASS=${CLASS}" >> ${FILE}
    date +'%Y-%m-%d-%H-%M-%S' >> ${FILE}
    CUDA_VISIBLE_DEVICES=0 TMPDIR=/data/bertoldo/tmp python runners/run_fmnist.py --it 1 --cls-restrictions ${CLASS}
    date +'%Y-%m-%d-%H-%M-%S' >> ${FILE}
    echo "" >> ${FILE}
done

###############################################################
FILE="cifar10.log"
CLASSES_LIST=( $( seq 0 9 ) )

echo "launching cifar10"
touch ${FILE}

for CLASS in "${CLASSES_LIST[@]}"
do
    echo "CLASS=${CLASS}" >> ${FILE}
    date +'%Y-%m-%d-%H-%M-%S' >> ${FILE}
    CUDA_VISIBLE_DEVICES=0 TMPDIR=/data/bertoldo/tmp python runners/run_cifar10.py --it 1 --cls-restrictions ${CLASS}
    date +'%Y-%m-%d-%H-%M-%S' >> ${FILE}
    echo "" >> ${FILE}
done


# i dont need this, the training time is already in the log.txt

