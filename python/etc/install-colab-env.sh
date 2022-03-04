#!/usr/bin/env bash
set -e  # stops the execution of a script if a command or pipeline has an

# install the conda env fcdd_rc21

FCDD_PYTHON_DIR=/content/drive/MyDrive/fcdd/python

echo "going to fcdd/python folder"
cd ${FCDD_PYTHON_DIR}
echo "pwd=$(pwd)"

echo "installing fcdd with pip install --editable"
pip install --editable .
# pip install --requirement requirements.txt

echo "installing colab dependencies"
pip install --requirement requirements-colab.txt

echo "installing the ipykernel locally (--user)"
python -m ipykernel install --user --name=fcdd
