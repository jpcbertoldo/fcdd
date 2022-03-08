#!/usr/bin/env bash
set -e  # stops the execution of a script if a command or pipeline has an

# install the conda env fcdd_dima2022

source ${HOME}/init-conda-bash
FCDD_PYTHON_DIR=${HOME}/fcdd/python
conda env create --file ${FCDD_PYTHON_DIR}/etc/condaenv/fcdd_dima2022.yml
conda activate fcdd_dima2022
cd ${FCDD_PYTHON_DIR}
pip install --editable .
conda activate fcdd_dima2022
echo "installing the ipykernel locally (--user)"
python -m ipykernel install --user --name=fcdd_dima2022