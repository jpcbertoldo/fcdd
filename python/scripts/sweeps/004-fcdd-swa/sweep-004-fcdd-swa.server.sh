#!/usr/bin/env bash
set -e  # stops the execution of a script if a command or pipeline has an 

source ${HOME}/init-conda-bash
conda activate fcdd_rc21

cd ${HOME}/fcdd/python/scripts/sweeps/004-fcdd-swa

wandb sweep --entity mines-paristech-cmm --project fcdd-mvtec-bmvc-01 --verbose --name sweep-04-fcdd-swa sweep-004-fcdd-swa.yaml

# sweep id: n0pdwtp5
# full id: mines-paristech-cmm/fcdd-mvtec-bmvc-01/n0pdwtp5