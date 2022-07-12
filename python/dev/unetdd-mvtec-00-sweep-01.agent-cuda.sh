#!/usr/bin/env bash
set -e  # stops the execution of a script if a command or pipeline has an 

source ${HOME}/init-conda-bash
conda activate fcdd_rc21
cd ${HOME}/fcdd/python/dev

# server
# wandb sweep --project unetdd-mvtec-00 --entity mines-paristech-cmm --verbose --name sweep-01 unetdd-mvtec-00-sweep-01.yaml

# agent 
export CUDA_VISIBLE_DEVICES=$1 

if [ -z "$CUDA_VISIBLE_DEVICES" ] || [ "$CUDA_VISIBLE_DEVICES" -lt 0 ] || [ "$CUDA_VISIBLE_DEVICES" -gt 3 ]
then
    echo "CUDA_VISIBLE_DEVICES (\$1) must be between 0 and 3, found: $CUDA_VISIBLE_DEVICES"
    exit 1
fi


# OLD (IGNORED)
# wandb agent mines-paristech-cmm/unetdd-mvtec-00/8g7n2cc5

wandb agent mines-paristech-cmm/unetdd-mvtec-00/7e6qg2bj