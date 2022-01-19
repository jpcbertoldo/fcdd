#!/usr/bin/env bash
set -e  # stops the execution of a script if a command or pipeline has an

# install the conda env fcdd_rc21

source ${HOME}/init-conda-bash
FCDD_PYTHON_DIR=${HOME}/fcdd/python
conda env create --file ${FCDD_PYTHON_DIR}/etc/condaenv/fcdd_rc21.yml
conda activate fcdd_rc21
cd ${FCDD_PYTHON_DIR}
pip install --editable .

