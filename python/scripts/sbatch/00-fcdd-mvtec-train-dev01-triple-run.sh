#! /bin/bash
 
# Personalisation de la tache
#
#SBATCH --job-name   fcdd-mvtec-train-dev01-triple-run
 
#SBATCH --partition  cmm-gpu
#SBATCH --gres       gpu:1
# SBATCH --nodelist   node002

#SBATCH --output     /cluster/CMM/home/jcasagrandebertoldo/log/fcdd/mvtec/train-dev01/%x-%N-%j.log
 
# tous les evenements pertinents seront envoyes par email a cette adresse
#SBATCH --mail-type  ALL
#SBATCH --mail-user joaopcbertoldo@gmail.com

# obs : la ligne suivante est necessaire pour forces l'exécution 
. $HOME/.bashrc

# ==============================================================================
# ================================== VARIABLES =================================
# ==============================================================================

Node=$(hostname)

# exec
WorkDir=$HOME/fcdd/python/dev
CONDA_ENV_NAME="fcdd_rc21"
SCRIPT_FNAME="train_mvtec_dev01.py"

# etc
MYTMPDIR="/cluster/CMM/data1/jcasagrandebertoldo/tmp"

# logging
LOGDIR=${HOME}/log/fcdd/mvtec/train-dev01
JNAME_BASE="${SLURM_JOB_NAME}-${Node}-${SLURM_JOB_ID}.python-print"
JPATH_BASE=${LOGDIR}/$JNAME_BASE
JPATH_00=${JPATH_BASE}.launch-00.log    
JPATH_01=${JPATH_BASE}.launch-01.log    
JPATH_02=${JPATH_BASE}.launch-02.log


echo "\$Node = $Node"

echo "\$WorkDir = $WorkDir"
echo "\$CONDA_ENV_NAME = $CONDA_ENV_NAME"
echo "\$SCRIPT_FNAME = $SCRIPT_FNAME"

echo "\$MYTMPDIR = ${MYTMPDIR}"

echo "\${LOGDIR} = ${LOGDIR}"
echo "\$JNAME_BASE = $JNAME_BASE"
echo "\$JPATH_BASE = $JPATH_BASE"   
echo "\$JPATH_00 = $JPATH_00.txt"
echo "\$JPATH_01 = $JPATH_01.txt"
echo "\$JPATH_02 = $JPATH_02.txt"

# ==============================================================================
# ==================================== SETUP ===================================
# ==============================================================================

echo "going to WorkDir"
cd $WorkDir

echo "loading conda"
source ${HOME}/init-conda-bash

echo "activating conda env"
conda activate $CONDA_ENV_NAME

echo "creating log dir"
mkdir -p ${LOGDIR}

echo "changing tmp dir"
export TMPDIR=$MYTMPDIR

# ==============================================================================
# ================================ HEALTH CHECKS ===============================
# ==============================================================================

echo "health checks"
echo "pwd = $(pwd)"
echo "which conda = $(which conda)"
echo "\$CONDA_DEFAULT_ENV = $CONDA_DEFAULT_ENV"
echo "which python = $(which python)"
echo "\$TMPDIR = ${TMPDIR}"
echo "nvcc --version = $(nvcc --version)"
echo "nvidia-smi"
nvidia-smi

# ==============================================================================
# ==================================== ARGS ====================================
# ==============================================================================

COMMON_ARGS="--wandb-tags dev01 gpu-triple-run"

ALL_COMMON_ARGS="${COMMON_ARGS} $*"

ARGS00="${ALL_COMMON_ARGS}"
ARGS01="${ALL_COMMON_ARGS}"
ARGS02="${ALL_COMMON_ARGS}"

echo ""
echo ""
echo "================ ARGS ================"
echo ""
echo ""
echo "\$COMMON_ARGS = ${COMMON_ARGS}"
echo "\$* = $*"
echo "\$ALL_COMMON_ARGS = ${ALL_COMMON_ARGS}"
echo "\$ARGS00 = ${ARGS00}"
echo "\$ARGS01 = ${ARGS01}"
echo "\$ARGS02 = ${ARGS02}"

# ==============================================================================
# ==================================== LAUNCH ==================================
# ==============================================================================

# & will put it in the background

echo "launching process 00"
python ${SCRIPT_FNAME} ${ARGS00} > "${JPATH_00}" 2>&1 & 
PYTHON_PID00=$!
echo "PYTHON_PID00=${PYTHON_PID00}"
sleep 7

echo "launching process 01"
python ${SCRIPT_FNAME} ${ARGS01} > "${JPATH_01}" 2>&1 & 
PYTHON_PID01=$!
echo "PYTHON_PID01=${PYTHON_PID01}"
sleep 7

# & will put it in the background
echo "launching process 02"
python ${SCRIPT_FNAME} ${ARGS02} > "${JPATH_02}" 2>&1 & 
PYTHON_PID02=$!
echo "PYTHON_PID02=${PYTHON_PID02}"
sleep 7

# ==============================================================================
# ====================================== WAIT ==================================
# ==============================================================================

echo "waiting for PYTHON_PID00=${PYTHON_PID00}"
wait ${PYTHON_PID00}
echo "PYTHON_PID00=${PYTHON_PID00} finished"

echo "waiting for PYTHON_PID01=${PYTHON_PID01}"
wait ${PYTHON_PID01}
echo "PYTHON_PID01=${PYTHON_PID01} finished"

echo "waiting for PYTHON_PID02=${PYTHON_PID02}"
wait ${PYTHON_PID02}
echo "PYTHON_PID02=${PYTHON_PID02} finished"

