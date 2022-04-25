#! /bin/bash
 
# Personalisation de la tache
#
#SBATCH --job-name   fcdd-mvtec
 
#SBATCH --partition  gpu-cmm
#SBATCH --gres       gpu:1
#SBATCH --mem        48G

#SBATCH --output     log/fcdd/mvtec/%N/%x-%N-%j.log
 
# tous les evenements pertinents seront envoyes par email a cette adresse
#SBATCH --mail-type  ALL
#SBATCH --mail-user joaopcbertoldo@gmail.com
 
# obs : la ligne suivante est necessaire pour forces l'exécution 
. $HOME/.bashrc
 
# Decomenter et definir le repertoire de travail
WorkDir=$HOME/fcdd/python/fcdd
cd $WorkDir

echo ""
echo "WorkDir = $WorkDir"
echo "pwd = $(pwd)"

Node=$(hostname)
echo "hostname = $(hostname)"

LOGDIR=${HOME}/log/fcdd/mvtec/${Node}
mkdir -p ${LOGDIR}
echo "LOGDIR = ${LOGDIR}"
 
source ${HOME}/bashrc.d/90-init-module.sh
echo $(ls -l ${HOME}/bashrc.d/90-init-module.sh)
echo "module = $(type -a module)"

# ajoute modules CUDA
module add cuda90
module add cuda90/blas
module add cuda90/toolkit
 
# use my local conda
source ${HOME}/init-conda-bash

echo ""
echo "which conda = $(which conda)"

# the env here is not fcdd_rc21 because of compatibility issues in thalassa
# see fcdd/python/etc/condaenv/readme.md
conda activate fcdd_rc21_torch181

echo ""
echo "\$CONDA_DEFAULT_ENV = $CONDA_DEFAULT_ENV"

echo ""
echo "nvidia-smi"
nvidia-smi

echo ""
echo "nvcc --version = $(nvcc --version)"

JNAME="${SLURM_JOB_NAME}-${Node}-${SLURM_JOB_ID}.print.log"
LOG_MEM_NAME="${SLURM_JOB_NAME}-${Node}-${SLURM_JOB_ID}.mem.log"
LOG_GPU_NAME="${SLURM_JOB_NAME}-${Node}-${SLURM_JOB_ID}.gpu.log"

echo ""
echo "\$JNAME = $JNAME"
echo "\$LOG_MEM_NAME = $LOG_MEM_NAME"
echo "\$LOG_GPU_NAME = $LOG_GPU_NAME"

JPATH=${LOGDIR}/$JNAME
LOG_MEM_FPATH=${LOGDIR}/${LOG_MEM_NAME}
LOG_GPU_FPATH=${LOGDIR}/${LOG_GPU_NAME}

echo ""
echo "\$JPATH = $JPATH"
echo "\$LOG_MEM_FPATH = $LOG_MEM_FPATH"
echo "\$LOG_GPU_FPATH = $LOG_GPU_FPATH"

echo ""
echo "which python = $(which python)"

export TMPDIR=/mnt/data2/CMM/jpcasagrande/tmp

echo ""
echo "\$TMPDIR = ${TMPDIR}"

echo ""
echo "\$* = $*"

# & will put it in the background, so I can go ahead and launch the memory monitor
python runners/run_mvtec.py $* > $JPATH 2>&1 & 
PYTHON_PID=$!

echo ""
echo "PYTHON_PID=${PYTHON_PID}"

echo ""
echo "launching mem monitor on pid ${PYTHON_PID}"
${HOME}/.local/bin/pid-monitor --pid ${PYTHON_PID} --dt 1 --file ${LOG_MEM_FPATH} &
MONITOR_MEM_PID=$!

echo ""
echo "MONITOR_MEM_PID=${MONITOR_MEM_PID}"

# gpustat not working!!!!!!!
# echo "launching gpu monitor"
# ${HOME}/.local/bin/gpustat --show-cmd --show-user --show-pid --interval 1 >> ${LOG_GPU_FPATH} &
# MONITOR_GPU_PID=$!
# echo ""
# echo "MONITOR_GPU_PID=${MONITOR_GPU_PID}"

####################################################
echo ""
echo "waiting for PYTHON_PID=${PYTHON_PID}"
wait ${PYTHON_PID}
echo "PYTHON_PID=${PYTHON_PID} finished"

echo ""
echo "killing MONITOR_MEM_PID=${MONITOR_MEM_PID}"
kill -9 ${MONITOR_MEM_PID}

# gpustat not working!!!!!!!
# echo ""
# echo "killing MONITOR_GPU_PID=${MONITOR_GPU_PID}"
# kill -9 ${MONITOR_GPU_PID}
