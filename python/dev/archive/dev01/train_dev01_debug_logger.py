#!/usr/bin/env python
# coding: utf-8
import contextlib
import functools
import os
import time
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from re import A
from typing import Any, Callable, List, Optional, Tuple

import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.profiler import (AdvancedProfiler, PyTorchProfiler,
                                        SimpleProfiler)
from pytorch_lightning.trainer.states import RunningStage
from torch.profiler import tensorboard_trace_handler

__version__ = "v1.0.0"


import wandb

logdir = Path.home() / "tmp"
            
wandb_init_kwargs = dict(
    project="mvtec-debug", 
    # name="debug-logger",
    entity="mines-paristech-cmm",
    tags=["debug-logger-minimal"],
    config=dict(),
    save_code=True,
    reinit=True,
)
wandb_logger = WandbLogger(
    save_dir=str(logdir),
    offline=True,
    # for now only the last checkpoint is available, but later the others can be integrated
    # (more stuff have to be done in the run_one())
    log_model=False,
    **wandb_init_kwargs,
)   
    
# image logging is not working properly with the logger
# so i also use the default wandb interface for that
wandb.init(
    dir=str(logdir),  # equivalent of savedir in the logger
    **{
        **wandb_init_kwargs,
        # make sure both have the same run_idimg_batch.clone()
        **dict(id=wandb_logger.experiment._run_id),  
    },
)
import time
time.sleep(3)
wandb.finish(0)
