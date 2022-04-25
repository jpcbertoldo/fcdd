#! /bin/bash
 
# Personalisation de la tache
#
#SBATCH --job-name   fcdd
 
#SBATCH --partition  gpu-cmm
#SBATCH --gres       gpu:1
#SBATCH --mem        96G

#SBATCH --output     log/%x-%N-%j.log
 
# tous les evenements pertinents seront envoyes par email a cette adresse
#SBATCH --mail-type  ALL
#SBATCH --mail-user joaopcbertoldo@gmail.com
 
# obs : la ligne suivante est necessaire pour forces l'exécution 
. $HOME/.bashrc
 
# Decomenter et definir le repertoire de travail
WorkDir=$HOME/fcdd/python/fcdd
cd $WorkDir

echo ""
echo "pwd = $(pwd)"

mkdir -p log
 
Node=$(hostname)
 
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

JNAME="${SLURM_JOB_NAME}-${SLURM_JOB_ID}"

echo ""
echo "\$JNAME = $JNAME"

echo ""
echo "which python = $(which python)"

export TMPDIR=/mnt/data2/CMM/jpcasagrande/tmp

echo ""
echo "\$TMPDIR = ${TMPDIR}"


echo ""
echo "\$* = $*"

python $* > $HOME/log/$JNAME.out 2>&1