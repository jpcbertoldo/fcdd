#!/usr/bin/env bash
set -e  # stops the execution of a script if a command or pipeline has an

# install the conda env unetdd

source ${HOME}/init-conda-bash
FCDD_PYTHON_DIR=${HOME}/fcdd/python

conda env create --file ${FCDD_PYTHON_DIR}/etc/condaenv/unetdd.yml
conda activate unetdd

# mmsegmentation
# pip install mmcv-full  # didnt work on lupi
cd ${FCDD_PYTHON_DIR}

pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cpu/torch1.10.0/index.html

# pip install mmsegmentation
git submodule add https://github.com/open-mmlab/mmsegmentation.git
cd mmsegmentation
pip install --editable .
cd ..

cd ${FCDD_PYTHON_DIR}
pip install --editable .

echo "installing the ipykernel locally (--user)"
python -m ipykernel install --user --name=unetdd