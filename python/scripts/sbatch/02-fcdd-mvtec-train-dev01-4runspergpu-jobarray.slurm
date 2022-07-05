#! /bin/bash
 
# Personalisation de la tache

#SBATCH --job-name   fcdd-mvtec-train-dev01-4runspergpu-jobarray
#SBATCH --partition  cmm-gpu
#SBATCH --gres       gpu:1
#SBATCH --cpus-per-task 2

# SLURM_ARRAY_TASK_ID \in {0, 4, 8, 12}, at idx >= 15 the script will exit
#SBATCH --array=0-15:4%1   # Id of job start-end % number of simultaneous running job

# variable meanings
# %x = job-name
# %N = node-name
# %A = SLURM_ARRAY_JOB_ID (same for all jobs in the same array)
# %a = SLURM_ARRAY_TASK_ID (the job index in the array)
# %j = SLURM_JOBID (unique for each job)
#SBATCH --output     /cluster/CMM/home/jcasagrandebertoldo/log/fcdd/mvtec/train-dev01-4runspergpu/%x-%N-%A_%a-%j.log
 
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
SLURM_SCRIPT_FNAME=`basename "$0"`

# etc
MYTMPDIR="/cluster/CMM/data1/jcasagrandebertoldo/tmp"

# logging
LOGDIR="${HOME}/log/fcdd/mvtec/train-dev01-4runspergpu"
JNAME_BASE="${SLURM_JOB_NAME}-${Node}-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}-${SLURM_JOB_ID}.python-print"
JPATH_BASE=${LOGDIR}/$JNAME_BASE
JPATH_00=${JPATH_BASE}.launch-00.log    
JPATH_01=${JPATH_BASE}.launch-01.log    
JPATH_02=${JPATH_BASE}.launch-02.log
JPATH_03=${JPATH_BASE}.launch-03.log


echo "\$Node = $Node"

echo "\$WorkDir = $WorkDir"
echo "\$CONDA_ENV_NAME = $CONDA_ENV_NAME"
echo "\$SCRIPT_FNAME = $SCRIPT_FNAME"
echo "\$SLURM_SCRIPT_FNAME = $SLURM_SCRIPT_FNAME"

echo "\$MYTMPDIR = ${MYTMPDIR}"

echo "\${LOGDIR} = ${LOGDIR}"
echo "\$JNAME_BASE = $JNAME_BASE"
echo "\$JPATH_BASE = $JPATH_BASE"   
echo "\$JPATH_00 = $JPATH_00.txt"
echo "\$JPATH_01 = $JPATH_01.txt"
echo "\$JPATH_02 = $JPATH_02.txt"
echo "\$JPATH_03 = $JPATH_03.txt"

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

# SLURM_ARRAY_TASK_ID \in {0, 2, ..., 30}
echo "SLURM_ARRAY_TASK_ID = $SLURM_ARRAY_TASK_ID"

if (( $SLURM_ARRAY_TASK_ID >= 15 ))
then
    echo "SLURM_ARRAY_TASK_ID >= 30"
    echo "exiting"
    exit
fi

COMMON_ARGS="${COMMON_ARGS} --nworkers 1"
COMMON_ARGS="${COMMON_ARGS} --wandb-tags dev01 4runspergpu"
ALL_COMMON_ARGS="${COMMON_ARGS} $*"

ARGS00="${ALL_COMMON_ARGS} --classes $(( $SLURM_ARRAY_TASK_ID + 0 ))"
ARGS01="${ALL_COMMON_ARGS} --classes $(( $SLURM_ARRAY_TASK_ID + 1 ))"
ARGS02="${ALL_COMMON_ARGS} --classes $(( $SLURM_ARRAY_TASK_ID + 2 ))"
ARGS03="${ALL_COMMON_ARGS} --classes $(( $SLURM_ARRAY_TASK_ID + 3 ))"

echo ""
echo ""
echo "================ ARGS ================"
echo ""
echo ""
echo "\$COMMON_ARGS = ${COMMON_ARGS}"
echo "\$TASK_ARGS = ${TASK_ARGS}"
echo "\$* = $*"
echo "\$ALL_COMMON_ARGS = ${ALL_COMMON_ARGS}"
echo "\$ARGS00 = ${ARGS00}"
echo "\$ARGS01 = ${ARGS01}"
echo "\$ARGS02 = ${ARGS02}"
echo "\$ARGS03 = ${ARGS03}"

# ==============================================================================
# ==================================== LAUNCH ==================================
# ==============================================================================

# & will put it in the background
echo "launching process 00"
python ${SCRIPT_FNAME} ${ARGS00} > "${JPATH_00}" 2>&1 & 
PYTHON_PID00=$!
echo "PYTHON_PID00=${PYTHON_PID00}"
sleep 7

# & will put it in the background
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

# & will put it in the background
echo "launching process 03"
python ${SCRIPT_FNAME} ${ARGS03} > "${JPATH_03}" 2>&1 & 
PYTHON_PID03=$!
echo "PYTHON_PID03=${PYTHON_PID03}"
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

echo "waiting for PYTHON_PID03=${PYTHON_PID03}"
wait ${PYTHON_PID03}
echo "PYTHON_PID03=${PYTHON_PID03} finished"

