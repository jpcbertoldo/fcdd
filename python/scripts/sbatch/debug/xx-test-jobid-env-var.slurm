#! /bin/bash
 
# Personalisation de la tache

#SBATCH --job-name   test-jobid-env-var
#SBATCH --partition  cmm-gpu
#SBATCH --gres       gpu:1
#SBATCH --cpus-per-task 1

# variable meanings
# %x = job-name
# %N = node-name
# %A = SLURM_ARRAY_JOB_ID (same for all jobs in the same array)
# %a = SLURM_ARRAY_TASK_ID (the job index in the array)
# %j = SLURM_JOBID (unique for each job)
#SBATCH --output     /cluster/CMM/home/jcasagrandebertoldo/log/debug/%x-%N-%A_%a-%j.log
 
# tous les evenements pertinents seront envoyes par email a cette adresse
#SBATCH --mail-type  ALL
#SBATCH --mail-user joaopcbertoldo@gmail.com

# SBATCH --array=0-3%1   # Id of job start-end % number of simultaneous running job

# obs : la ligne suivante est necessaire pour forces l'exécution 
. $HOME/.bashrc

# ==============================================================================
# ================================== VARIABLES =================================
# ==============================================================================

Node=$(hostname)

# exec
WorkDir=$HOME/fcdd/python/scripts/sbatch/debug
CONDA_ENV_NAME="fcdd_rc21"
SCRIPT_FNAME="test_jobid_env_var.py"

# logging
LOGDIR="${HOME}/log/debug"
JNAME_BASE="${SLURM_JOB_NAME}-${Node}-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}-${SLURM_JOB_ID}.python-print"
JPATH_BASE=${LOGDIR}/$JNAME_BASE
JPATH_00=${JPATH_BASE}.launch-00.log

echo "\$Node = $Node"

echo "\$WorkDir = $WorkDir"
echo "\$CONDA_ENV_NAME = $CONDA_ENV_NAME"
echo "\$SCRIPT_FNAME = $SCRIPT_FNAME"

echo "\${LOGDIR} = ${LOGDIR}"
echo "\$JNAME_BASE = $JNAME_BASE"
echo "\$JPATH_BASE = $JPATH_BASE"   
echo "\$JPATH_00 = $JPATH_00.txt"

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

# ==============================================================================
# ================================ HEALTH CHECKS ===============================
# ==============================================================================

echo "health checks"
echo "pwd = $(pwd)"
echo "which conda = $(which conda)"
echo "\$CONDA_DEFAULT_ENV = $CONDA_DEFAULT_ENV"
echo "\$SBATCH_OUTPUT = $SBATCH_OUTPUT"
echo "which python = $(which python)"

# ==============================================================================
# ==================================== ARGS ====================================
# ==============================================================================

ARGS00=""

echo "================ ARGS ================"
echo ""
echo ""
echo "\$ARGS00 = ${ARGS00}"
echo "\$* = $*"

# ==============================================================================
# ==================================== LAUNCH ==================================
# ==============================================================================

# & will put it in the background
echo "launching process 00"
python ${SCRIPT_FNAME} ${ARGS00} > "${JPATH_00}" 2>&1 & 
PYTHON_PID00=$!
echo "PYTHON_PID00=${PYTHON_PID00}"

# ==============================================================================
# ====================================== WAIT ==================================
# ==============================================================================

echo "waiting for PYTHON_PID00=${PYTHON_PID00}"
wait ${PYTHON_PID00}
echo "PYTHON_PID00=${PYTHON_PID00} finished"
