#!/usr/bin/env bash
set -e  # stops the execution of a script if a command or pipeline has an 

THIS_SCRIPT_DIR=$( cd ${0%/*} && pwd -P )
THIS_SCRIPT_DIR=$( realpath ${THIS_SCRIPT_DIR} )

LOGS_DIR="${HOME}/log/fcdd-memory-usage"
mkdir -p ${LOGS_DIR}
echo "LOGS_DIR=${LOGS_DIR}"

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

echo "pid-monitor=$(which pid-monitor)"

###############################################################
# DATASET="fmnist"
# RUNNER="run_fmnist.py"

# DATASET="cifar10"
# RUNNER="run_cifar10.py"

DATASET="mvtec-unsupervised"
RUNNER="run_mvtec.py"
# mvtec is unsupervised by default

# DATASET="mvtec-semi-supervised"
# RUNNER="run_mvtec.py"
# args for semi-supervised 
# --supervise-mode noise --noise-mode mvtec_gt --oe-limit 1

# DATASET="pascalvoc"
# RUNNER="run_pascalvoc.py"
###############################################################

LOG_MEM_FPATH="${LOGS_DIR}/${DATASET}.$(date +'%Y-%m-%d-%H-%M-%S').mem.log"
LOG_GPU_FPATH="${LOGS_DIR}/${DATASET}.$(date +'%Y-%m-%d-%H-%M-%S').gpu.log"

echo "launching ${DATASET}"
CUDA_VISIBLE_DEVICES=1 TMPDIR=/data/bertoldo/tmp python runners/${RUNNER} --it 1 --epochs 1 & 
# for mvtec semi-supervised
# CUDA_VISIBLE_DEVICES=0 TMPDIR=/data/bertoldo/tmp python runners/${RUNNER} --supervise-mode noise --noise-mode mvtec_gt --oe-limit 1 --it 1 --epochs 1 & 
PID=$!

echo "launching mem monitor on pid ${PID}"
pid-monitor --pid ${PID} --dt 1 --file ${LOG_MEM_FPATH} &
MONITOR_MEM_PID=$!

echo "launching gpu monitor"
gpustat --show-cmd --show-user --show-pid --interval 1 >> ${LOG_GPU_FPATH} &
MONITOR_GPU_PID=$!

echo "waiting for PID=${PID}"
wait ${PID}
echo "PID=${PID} finished"

echo "killing MONITOR_MEM_PID=${MONITOR_MEM_PID}"
kill -9 ${MONITOR_MEM_PID}

echo "killing MONITOR_GPU_PID=${MONITOR_GPU_PID}"
kill -9 ${MONITOR_GPU_PID}
