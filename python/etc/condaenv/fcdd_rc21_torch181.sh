#!/usr/bin/env bash
set -e  # stops the execution of a script if a command or pipeline has an

# install the conda env fcdd_rc21_torch181

source ${HOME}/init-conda-bash
FCDD_PYTHON_DIR=${HOME}/fcdd/python
conda env create --file ${FCDD_PYTHON_DIR}/etc/condaenv/fcdd_rc21_torch181.yml
conda activate fcdd_rc21_torch181
cd ${FCDD_PYTHON_DIR}
pip install --editable .

# downgrade torch
pip install --force-reinstall torch==1.8.1

# this gives the error 
# ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
# torchvision 0.10.1 requires torch==1.9.1, but you have torch 1.8.1 which is incompatible.
# fcdd 1.1.0 requires numpy==1.19.3, but you have numpy 1.22.1 which is incompatible.
# fcdd 1.1.0 requires torch==1.9.1, but you have torch 1.8.1 which is incompatible.

# i will ignore fcdd requirements and correct torchvision
# src: https://pypi.org/project/torchvision/#:~:text=1.8.1,%3E%3D3.6%2C%20%3C%3D3.9
pip install --force-reinstall torchvision==0.9.1

# and it gives 
# ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
# fcdd 1.1.0 requires numpy==1.19.3, but you have numpy 1.22.1 which is incompatible.
# fcdd 1.1.0 requires Pillow==8.3.2, but you have pillow 9.0.0 which is incompatible.
# fcdd 1.1.0 requires torch==1.9.1, but you have torch 1.8.1 which is incompatible.
# fcdd 1.1.0 requires torchvision==0.10.1, but you have torchvision 0.9.1 which is incompatible.

# let's try like this :)