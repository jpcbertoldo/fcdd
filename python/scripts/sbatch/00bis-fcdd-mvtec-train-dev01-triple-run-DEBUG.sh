#! /bin/bash
 
# Personalisation de la tache
#
#SBATCH --job-name   fcdd-mvtec-train-dev01-triple-run
 
#SBATCH --partition  cmm-gpu
#SBATCH --gres       gpu:1
#SBATCH --nodelist   node002

#SBATCH --output     /cluster/CMM/home/jcasagrandebertoldo/log/fcdd/mvtec/train-dev01-debug/%x-%N-%j.log
 
# tous les evenements pertinents seront envoyes par email a cette adresse
#SBATCH --mail-type  ALL
#SBATCH --mail-user joaopcbertoldo@gmail.com

 
# obs : la ligne suivante est necessaire pour forces l'exécution 
. $HOME/.bashrc
 
# Decomenter et definir le repertoire de travail
WorkDir=$HOME/fcdd/python/dev
cd $WorkDir

echo ""
echo "WorkDir = $WorkDir"
echo "pwd = $(pwd)"

Node=$(hostname)
echo "hostname = $(hostname)"

# use my local conda
source ${HOME}/init-conda-bash
echo ""
echo "which conda = $(which conda)"

# the env here is not fcdd_rc21 because of compatibility issues in thalassa
# see fcdd/python/etc/condaenv/readme.md
conda activate fcdd_rc21
echo ""
echo "\$CONDA_DEFAULT_ENV = $CONDA_DEFAULT_ENV"

echo ""
echo "nvidia-smi"
nvidia-smi

echo ""
echo "nvcc --version = $(nvcc --version)"

# ==============================================================================
# ==================================== SETUP ===================================
# ==============================================================================

LOGDIR=${HOME}/log/fcdd/mvtec/train-dev01-debug
mkdir -p ${LOGDIR}
echo "LOGDIR = ${LOGDIR}"
 
JNAME_BASE="${SLURM_JOB_NAME}-${Node}-${SLURM_JOB_ID}.python-print"
echo ""
echo "\$JNAME_BASE = $JNAME_BASE"

JPATH_BASE=${LOGDIR}/$JNAME_BASE
JPATH_00=${JPATH_BASE}.launch-00.log    
JPATH_01=${JPATH_BASE}.launch-01.log    
JPATH_02=${JPATH_BASE}.launch-02.log    
echo ""
echo "\$JPATH_BASE = $JPATH_BASE"
echo "\$JPATH_00 = $JPATH_00.txt"
echo "\$JPATH_01 = $JPATH_01.txt"
echo "\$JPATH_02 = $JPATH_02.txt"

echo ""
echo "which python = $(which python)"

export TMPDIR=/cluster/CMM/data1/jcasagrandebertoldo/tmp
echo ""
echo "\$TMPDIR = ${TMPDIR}"

# ==============================================================================
# ==================================== ARGS ====================================
# ==============================================================================

echo ""
echo "\$* = $*"

# DEBUG_ARGS_OFFLINE="--epochs 20 --wandb-offline --wandb-checkpoint-mode none"
# DEBUG_ARGS=$DEBUG_ARGS_OFFLINE

DEBUG_ARGS_ONLINE="--epochs 20"
DEBUG_ARGS=$DEBUG_ARGS_ONLINE

echo ""
echo "\$DEBUG_ARGS = ${DEBUG_ARGS}"

COMMON_ARGS="--n-seeds 1 ${DEBUG_ARGS}"
echo ""
echo "\$COMMON_ARGS = ${COMMON_ARGS}"


SCRIPT_FNAME="train_dev01_debug_logger.py"
echo ""
echo "\$SCRIPT_FNAME = $SCRIPT_FNAME"


# ==============================================================================
# ==================================== LAUNCH ==================================
# ==============================================================================

# & will put it in the background
echo ""
echo "launching process 00"
python ${SCRIPT_FNAME} ${COMMON_ARGS} $* > "${JPATH_00}" 2>&1 & 
PYTHON_PID00=$!
echo ""
echo "PYTHON_PID00=${PYTHON_PID00}"
sleep 7

# ==============================================================================
# ====================================== WAIT ==================================
# ==============================================================================

echo ""
echo "waiting for PYTHON_PID00=${PYTHON_PID00}"
wait ${PYTHON_PID00}
echo "PYTHON_PID00=${PYTHON_PID00} finished"