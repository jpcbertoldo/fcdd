#! /bin/bash
 
# Personalisation de la tache
#
#SBATCH --job-name   fcdd-mvtec-dev00-checkpoint02bis
 
#SBATCH --partition  cmm-gpu
#SBATCH --gres       gpu:1
#SBATCH --mem        32G
#SBATCH --nodelist   node002

#SBATCH --output     /cluster/CMM/home/jcasagrandebertoldo/log/fcdd/mvtec/dev00-checkpoint02bis/%x-%N-%j.log
 
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

LOGDIR=${HOME}/log/fcdd/mvtec/dev00-checkpoint02bis
mkdir -p ${LOGDIR}
echo "LOGDIR = ${LOGDIR}"
 
JNAME="${SLURM_JOB_NAME}-${Node}-${SLURM_JOB_ID}.python-print.log"
echo ""
echo "\$JNAME = $JNAME"

JPATH=${LOGDIR}/$JNAME
echo ""
echo "\$JPATH = $JPATH"

echo ""
echo "which python = $(which python)"

export TMPDIR=/cluster/CMM/data1/jcasagrandebertoldo/tmp
export WANDB="1"
echo ""
echo "\$TMPDIR = ${TMPDIR}"
echo "\$WANDB = ${WANDB}"

echo ""
echo "\$* = $*"




CLASSES_RESTRICTIONS_00="--cls-restrictions 0 1 2 3 4 5 6 7"
CLASSES_RESTRICTIONS_01="--cls-restrictions 8 9 10 11 12 13 14"


COMMON_ARGS="--it 3 --epochs 50"
echo ""
echo "\$COMMON_ARGS = ${COMMON_ARGS}"



SCRIPT_FNAME="train_mvtec_dev00_checkpoint02bis.py"
echo ""
echo "\$SCRIPT_FNAME = $SCRIPT_FNAME"




# & will put it in the background
echo ""
echo "launching process 00"
echo "\$CLASSES_RESTRICTIONS_00 = $CLASSES_RESTRICTIONS_00"
python ${SCRIPT_FNAME} ${COMMON_ARGS} $* $CLASSES_RESTRICTIONS_00 > "${JPATH}00" 2>&1 & 
PYTHON_PID00=$!
echo ""
echo "PYTHON_PID00=${PYTHON_PID00}"
sleep 7


# & will put it in the background
echo ""
echo "launching process 01"
echo "\$CLASSES_RESTRICTIONS_01 = $CLASSES_RESTRICTIONS_01"
python ${SCRIPT_FNAME} ${COMMON_ARGS} $* $CLASSES_RESTRICTIONS_01 > "${JPATH}01" 2>&1 & 
PYTHON_PID01=$!
echo ""
echo "PYTHON_PID01=${PYTHON_PID01}"
sleep 7




echo ""
echo "waiting for PYTHON_PID00=${PYTHON_PID00}"
wait ${PYTHON_PID00}
echo "PYTHON_PID00=${PYTHON_PID00} finished"

echo ""
echo "waiting for PYTHON_PID01=${PYTHON_PID01}"
wait ${PYTHON_PID01}
echo "PYTHON_PID01=${PYTHON_PID01} finished"
