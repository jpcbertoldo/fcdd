#!/usr/bin/env bash
set -e  # stops the execution of a script if a command or pipeline has an 

source ${HOME}/init-conda-bash
conda activate fcdd_rc21
cd ${HOME}/fcdd/python/scripts/sweeps/004-fcdd-swa

# agent 
export CUDA_VISIBLE_DEVICES=$1

if [ -z "$CUDA_VISIBLE_DEVICES" ] || [ "$CUDA_VISIBLE_DEVICES" -lt 0 ] || [ "$CUDA_VISIBLE_DEVICES" -gt 3 ]
then
    echo "CUDA_VISIBLE_DEVICES (\$1) must be between 0 and 3, found: $CUDA_VISIBLE_DEVICES"
    exit 1
fi

wandb agent mines-paristech-cmm/fcdd-mvtec-bmvc-01/ji05j3me