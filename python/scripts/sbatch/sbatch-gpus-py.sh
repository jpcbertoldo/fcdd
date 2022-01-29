#! /bin/bash
 
# Personalisation de la tache
#
#SBATCH --job-name   gpus-py
 
#SBATCH --partition  gpu-cmm
#SBATCH --gres       gpu:1
#SBATCH --mem        32768

#SBATCH --output     log/%x-%N-%j.log
 
# tous les evenements pertinents seront envoyes par email a cette adresse
#SBATCH --mail-type  ALL
#SBATCH --mail-user joaopcbertoldo@gmail.com
 
# obs : la ligne suivante est necessaire pour forces l'exécution 
. $HOME/.bashrc
 
# Decomenter et definir le repertoire de travail
WorkDir=$HOME/fcdd/python/fcdd
cd $WorkDir

echo "pwd = $(pwd)"

mkdir -p log
 
Node=$(hostname)
 
# ajoute modules CUDA
module add cuda90
module add cuda90/blas
module add cuda90/toolkit
 
# use my local conda
source ${HOME}/init-conda-bash

echo "which conda = $(which conda)"

conda activate fcdd_rc21_torch181

echo "\$CONDA_DEFAULT_ENV = $CONDA_DEFAULT_ENV"

echo "nvidia-smi"
nvidia-smi

echo ""
echo "nvcc --version = $(nvcc --version)"
echo ""

JNAME="${SLURM_JOB_NAME}-${SLURM_JOB_ID}"

echo "\$JNAME = $JNAME"
 
# code de calcul
# Noter 

echo "which python = $(which python)"

python ../scripts/gpus.py $* > $HOME/log/$JNAME.out 2>&1