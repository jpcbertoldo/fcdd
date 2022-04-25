#! /bin/bash
 
# Personalisation de la tache
#
#SBATCH --job-name   fcdd-mvtec-train-dev00
 
#SBATCH --partition  cmm-gpu
#SBATCH --gres       gpu:1
#SBATCH --mem        32G
#SBATCH --nodelist   node001

#SBATCH --output     /cluster/CMM/home/jcasagrandebertoldo/log/fcdd/mvtec/mvtec-train-dev00.checkpoint01/%x-%N-%j.log
 
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

# source ${HOME}/bashrc.d/90-init-module.sh
# echo $(ls -l ${HOME}/bashrc.d/90-init-module.sh)
# echo "module = $(type -a module)"

# # ajoute modules CUDA
# module add cuda90
# module add cuda90/blas
# module add cuda90/toolkit
 
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

LOGDIR=${HOME}/log/fcdd/mvtec/mvtec-train-dev00.checkpoint01
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

# & will put it in the background
python train_mvtec_dev00.py $* > $JPATH 2>&1 & 
PYTHON_PID=$!
echo ""
echo "PYTHON_PID=${PYTHON_PID}"
echo ""
echo "waiting for PYTHON_PID=${PYTHON_PID}"
wait ${PYTHON_PID}
echo "PYTHON_PID=${PYTHON_PID} finished"
