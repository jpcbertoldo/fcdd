#!/usr/bin/env bash
set -e  # stops the execution of a script if a command or pipeline has an 

source ${HOME}/init-conda-bash
conda activate fcdd_rc21
cd ${HOME}/fcdd/python/dev
wandb sweep --project unetdd-mvtec-00 --entity mines-paristech-cmm --verbose --name sweep-02 unetdd-mvtec-00-sweep-02.yaml


# OLD (IGNORED)
# sweep id: 8g7n2cc5
# agent line:
# wandb agent mines-paristech-cmm/unetdd-mvtec-00/8g7n2cc5
# full id: mines-paristech-cmm/unetdd-mvtec-00/8g7n2cc5

