#!/usr/bin/env bash
set -e  # stops the execution of a script if a command or pipeline has an 

source ${HOME}/init-conda-bash
conda activate fcdd_rc21

cd ${HOME}/fcdd/python/scripts/sweeps/003-fcdd-dropout-and-dataaug2

wandb sweep --entity mines-paristech-cmm --project fcdd-mvtec-bmvc-01 --verbose --name sweep-03 sweep-003-fcdd-dropout-and-dataaug2.yaml

# sweep id: b76erf8a
# full id: mines-paristech-cmm/fcdd-mvtec-bmvc-01/b76erf8a

