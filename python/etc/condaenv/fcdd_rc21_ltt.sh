#!/usr/bin/env bash
set -e  # stops the execution of a script if a command or pipeline has an

# install the conda env fcdd_rc21_ltt

source ${HOME}/init-conda-bash
FCDD_PYTHON_DIR=${HOME}/fcdd/python
conda env create --file ${FCDD_PYTHON_DIR}/etc/condaenv/fcdd_rc21_ltt.yml
conda activate fcdd_rc21_ltt
cd ${FCDD_PYTHON_DIR}

# FIRST ATTEMPT
#   # install ltt = light the torch
#   # this is an attempt to correct an error with cuda
#   # src: https://stackoverflow.com/a/65442492/9582881
#   # src[err]: https://docs.google.com/document/d/1cJWAM1gzHRG56qOtPXNsaw6uMCIfMyLqbdT6eBnaCvY/edit#bookmark=id.lz0uxpdhzjzw
#   pip install light-the-torch
#   
#   # versions copied from fcdd/python/requirements.txt
#   # torchvision==0.10.1
#   # torch==1.9.1
#   ltt install torch==1.9.1 torchvision==0.10.1
#   
#   pip install --editable .
#   
#   # didn't work, err:
#   # Traceback (most recent call last):
#   #   File "/cbio/donnees/jpcasagrande/repos/fcdd/python/fcdd/../scripts/gpus.py", line 7, in <module>
#   #     print(f"{(ngpus := cuda.device_count())=} {cuda.current_device()=}\n")
#   #   File "/cbio/donnees/jpcasagrande/miniconda3/envs/fcdd_rc21_ltt/lib/python3.9/site-packages/torch/cuda/__init__.py", line 432, in current_device
#   #     _lazy_init()
#   #   File "/cbio/donnees/jpcasagrande/miniconda3/envs/fcdd_rc21_ltt/lib/python3.9/site-packages/torch/cuda/__init__.py", line 166, in _lazy_init
#   #     raise AssertionError("Torch not compiled with CUDA enabled")
#   # AssertionError: Torch not compiled with CUDA enabled
#   
#   # firs, let me try this
#   conda install cudatoolkit=10.1 --channel pytorch --yes
#   
#   # same error
#   conda remove cudatoolkit --yes
#   
#   # let's try this?
#   # src: https://github.com/pmeier/light-the-torch#motivation
#   ltt install torch==1.9.1+cu101 torchvision==0.10.1+cu101
#   
#   # nope, err:
#   # Traceback (most recent call last):
#   #   File "/cbio/donnees/jpcasagrande/miniconda3/envs/fcdd_rc21_ltt/bin/ltt", line 8, in <module>
#   #     sys.exit(main())
#   #   File "/cbio/donnees/jpcasagrande/miniconda3/envs/fcdd_rc21_ltt/lib/python3.9/site-packages/light_the_torch/cli/__init__.py", line 12, in main
#   #     args = parse_args(args)
#   #   File "/cbio/donnees/jpcasagrande/miniconda3/envs/fcdd_rc21_ltt/lib/python3.9/site-packages/light_the_torch/cli/__init__.py", line 26, in parse_args
#   #     parser = make_ltt_parser()
#   #   File "/cbio/donnees/jpcasagrande/miniconda3/envs/fcdd_rc21_ltt/lib/python3.9/site-packages/light_the_torch/cli/__init__.py", line 87, in make_ltt_parser
#   #     add_ltt_find_parser(subparsers)
#   #   File "/cbio/donnees/jpcasagrande/miniconda3/envs/fcdd_rc21_ltt/lib/python3.9/site-packages/light_the_torch/cli/__init__.py", line 164, in add_ltt_find_parser
#   #     add_pip_install_arguments(parser, "platform", "python_version")
#   #   File "/cbio/donnees/jpcasagrande/miniconda3/envs/fcdd_rc21_ltt/lib/python3.9/site-packages/light_the_torch/cli/__init__.py", line 173, in add_pip_install_arguments
#   #     assert len(options) == 1
#   # AssertionError

#   # !!!!!! THIS IS NOT RELATED TO THE LINE ITSELF BECAUSE LTT STARTED DOING IT FOR EVERY COMMAND

# ------


# SECOND ATTEMPT
# repeting the attempt of 
# ltt install torch==1.9.1+cu101 torchvision==0.10.1+cu101
pip install light-the-torch
ltt install torch==1.9.1+cu101 torchvision==0.10.1+cu101
pip install --editable .
