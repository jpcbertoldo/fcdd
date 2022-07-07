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

import mvtec_dataset_dev01 as mvtec_dataset_dev01
import wandb
from callbacks_dev01 import (HEATMAP_NORMALIZATION_MINMAX_IN_EPOCH, HEATMAP_NORMALIZATION_PERCENTILES_IN_EPOCH, LOG_HISTOGRAM_MODES,
                             DataloaderPreviewCallback, LearningRateLoggerCallback,
                             LogAveragePrecisionCallback, LogHistogramCallback,
                             LogHistogramsSuperposedPerClassCallback,
                             LogImageHeatmapTableCallback,
                             LogPercentilesPerClassCallback,
                             LogPerInstanceValueCallback, LogRocCallback,
                             TorchTensorboardProfilerCallback)
from common_dev01 import (create_python_random_generator, create_seed,
                          hashify_config, seed_int2str, seed_str2int)

__version__ = "v1.0.1"


# when computing ROC on the pixels, extract a sample of this many pixels
# used in the ROC callbacks for train and validation
PIXELWISE_ROC_LIMIT_POINTS = 3000  # 3k

# when computing precision-recall curve (and avg precision) on the pixels, extract a sample of this many pixels
# used in the ROC callbacks for train and validation
PIXELWISE_PR_LIMIT_POINTS = 3000  # 3k

# when logging histograms, log the histograms of this many random values (not all in the array/tensor)
PIXELWISE_HISTOGRAMS_LIMIT_POINTS_TRAIN = 3000  # 3k
PIXELWISE_HISTOGRAMS_LIMIT_POINTS_VALIDATION = 3000  # 3k
PIXELWISE_HISTOGRAMS_LIMIT_POINTS_TEST = 30000  # 30k 


# ======================================== exceptions ========================================

class ScriptError(Exception):
    pass


class DatasetError(Exception):
    pass


class ModelError(Exception):
    pass


# ======================================== dataset ========================================

DATASET_CHOICES = (mvtec_dataset_dev01.DATASET_NAME,)
print(f"DATASET_CHOICES={DATASET_CHOICES}")


def unknown_dataset(wrapped: Callable[[str, ], Any]):
    @functools.wraps(wrapped)
    def wrapper(dataset: str, *args, **kwargs) -> Any:
        assert dataset in DATASET_CHOICES
        return wrapped(dataset, *args, **kwargs)
    return wrapper

@unknown_dataset
def dataset_class_labels(dataset_name: str) -> List[str]:
    return {
        mvtec_dataset_dev01.DATASET_NAME: mvtec_dataset_dev01.CLASSES_LABELS,
    }[dataset_name]
    

@unknown_dataset
def dataset_class_fullqualified(dataset_name: str) -> List[str]:
    return {
        mvtec_dataset_dev01.DATASET_NAME: mvtec_dataset_dev01.CLASSES_FULLQUALIFIED,
    }[dataset_name]


@unknown_dataset
def dataset_nclasses(dataset_name: str) -> int:
    return {
        mvtec_dataset_dev01.DATASET_NAME: mvtec_dataset_dev01.NCLASSES,
    }[dataset_name]


@unknown_dataset
def dataset_class_index(dataset_name: str, class_name: str) -> int:
    return dataset_class_labels(dataset_name).index(class_name)


# ======================================== preprocessing ========================================

@unknown_dataset
def dataset_preprocessing_choices(dataset_name: str) -> List[str]:
    return {
        mvtec_dataset_dev01.DATASET_NAME: mvtec_dataset_dev01.PREPROCESSING_CHOICES,
    }[dataset_name]


ALL_PREPROCESSING_CHOICES = tuple(set.union(*[
    set(dataset_preprocessing_choices(dataset_name)) 
    for dataset_name in DATASET_CHOICES
]))
print(f"PREPROCESSING_CHOICES={ALL_PREPROCESSING_CHOICES}")


# ======================================== supervise mode ========================================

@unknown_dataset
def dataset_supervise_mode_choices(dataset_name: str) -> List[str]:
    return {
        mvtec_dataset_dev01.DATASET_NAME: mvtec_dataset_dev01.SUPERVISE_MODES,
    }[dataset_name]


ALL_SUPERVISE_MODE_CHOICES = tuple(set.union(*[
    set(dataset_supervise_mode_choices(dataset_name)) 
    for dataset_name in DATASET_CHOICES
]))
print(f"SUPERVISE_MODE_CHOICES={ALL_SUPERVISE_MODE_CHOICES}")


# ======================================== models ========================================

import model_dev01

MODEL_CLASSES = {
    model_dev01.MODEL_FCDD_CNN224_VGG_F: model_dev01.FCDD,
    model_dev01.MODEL_U2NET_HEIGHT4_LITE: model_dev01.HyperSphereU2Net,
    model_dev01.MODEL_U2NET_HEIGHT6_LITE: model_dev01.HyperSphereU2Net,
    model_dev01.MODEL_U2NET_HEIGHT6_FULL: model_dev01.HyperSphereU2Net,
    model_dev01.MODEL_U2NET_REPVGG_HEIGHT4_LITE: model_dev01.HyperSphereU2Net,
    model_dev01.MODEL_U2NET_REPVGG_HEIGHT6_FULL: model_dev01.HyperSphereU2Net,
}
MODEL_CHOICES = tuple(sorted(MODEL_CLASSES.keys()))
print(f"MODEL_CHOICES={MODEL_CHOICES}")

def unknown_model(wrapped: Callable[[str, ], Any]):
    @functools.wraps(wrapped)
    def wrapper(model: str, *args, **kwargs) -> Any:
        assert model in MODEL_CHOICES
        return wrapped(model, *args, **kwargs)
    return wrapper

# ======================================== optmizers ========================================

@unknown_model
def model_optimizer_choices(model_name: str) -> List[str]:
    return {
        model_dev01.MODEL_FCDD_CNN224_VGG_F: model_dev01.OPTIMIZER_CHOICES,
        model_dev01.MODEL_U2NET_HEIGHT4_LITE: model_dev01.OPTIMIZER_CHOICES,
        model_dev01.MODEL_U2NET_HEIGHT6_LITE: model_dev01.OPTIMIZER_CHOICES,
        model_dev01.MODEL_U2NET_HEIGHT6_FULL: model_dev01.OPTIMIZER_CHOICES,
        model_dev01.MODEL_U2NET_REPVGG_HEIGHT4_LITE: model_dev01.OPTIMIZER_CHOICES,
        model_dev01.MODEL_U2NET_REPVGG_HEIGHT6_FULL: model_dev01.OPTIMIZER_CHOICES,
    }[model_name]   


OPTIMIZER_CHOICES = tuple(set.union(*[
    set(model_optimizer_choices(model_name))
    for model_name in MODEL_CHOICES
]))
print(f"OPTIMIZER_CHOICES={OPTIMIZER_CHOICES}")  

# ======================================== schedulers ========================================

@unknown_model
def model_scheduler_choices(model_name: str) -> List[str]:
    return {
        model_dev01.MODEL_FCDD_CNN224_VGG_F: model_dev01.SCHEDULER_CHOICES,
        model_dev01.MODEL_U2NET_HEIGHT4_LITE: model_dev01.SCHEDULER_CHOICES,
        model_dev01.MODEL_U2NET_HEIGHT6_LITE: model_dev01.SCHEDULER_CHOICES,
        model_dev01.MODEL_U2NET_HEIGHT6_FULL: model_dev01.SCHEDULER_CHOICES,
        model_dev01.MODEL_U2NET_REPVGG_HEIGHT4_LITE: model_dev01.SCHEDULER_CHOICES,
        model_dev01.MODEL_U2NET_REPVGG_HEIGHT6_FULL: model_dev01.SCHEDULER_CHOICES,
    }[model_name]


SCHEDULER_CHOICES = tuple(set.union(*[
    set(model_scheduler_choices(model_name))
    for model_name in MODEL_CHOICES
]))
print(f"SCHEDULER_CHOICES={SCHEDULER_CHOICES}")

# ======================================== losses ========================================

@unknown_model
def model_loss_choices(model_name: str) -> List[str]:
    return {
        model_dev01.MODEL_FCDD_CNN224_VGG_F: model_dev01.LOSS_FCDD_CHOICES,
        model_dev01.MODEL_U2NET_HEIGHT4_LITE: model_dev01.LOSS_U2NET_CHOICES,
        model_dev01.MODEL_U2NET_HEIGHT6_LITE: model_dev01.LOSS_U2NET_CHOICES,
        model_dev01.MODEL_U2NET_HEIGHT6_FULL: model_dev01.LOSS_U2NET_CHOICES,
        model_dev01.MODEL_U2NET_REPVGG_HEIGHT4_LITE: model_dev01.LOSS_U2NET_CHOICES,
        model_dev01.MODEL_U2NET_REPVGG_HEIGHT6_FULL: model_dev01.LOSS_U2NET_CHOICES,
    }[model_name]


LOSS_CHOICES = tuple(set.union(*[
    set(model_loss_choices(model_name))
    for model_name in MODEL_CHOICES
]))
print(f"LOSS_CHOICES={LOSS_CHOICES}")


# ======================================== pytorch lightning ========================================

LIGHTNING_ACCELERATOR_CPU = "cpu"
LIGHTNING_ACCELERATOR_GPU = "gpu"
LIGHTNING_ACCELERATOR_CHOICES = (
    LIGHTNING_ACCELERATOR_CPU, 
    LIGHTNING_ACCELERATOR_GPU,
)
print(f"LIGHTNING_ACCELERATOR_CHOICES={LIGHTNING_ACCELERATOR_CHOICES}")

# src: https://pytorch-lightning.readthedocs.io/en/latest/extensions/strategy.html
LIGHTNING_STRATEGY_DDP = "ddp"
LIGHTNING_STRATEGY_CHOICES = (
    None,
    LIGHTNING_STRATEGY_DDP,
)
print(f"LIGHTNING_STRATEGY_CHOICES={LIGHTNING_STRATEGY_CHOICES}")


LIGHTNING_PRECISION_32 = 32
LIGHTNING_PRECISION_16 = 16
LIGHTNING_PRECISION_CHOICES = (
    LIGHTNING_PRECISION_32,
    LIGHTNING_PRECISION_16,
)
print(f"LIGHTNING_PRECISION_CHOICES={LIGHTNING_PRECISION_CHOICES}")


LIGHTNING_PROFILER_SIMPLE = "simple"
LIGHTNING_PROFILER_ADVANCED = "advanced"
LIGHTNING_PROFILER_PYTORCH = "pytorch"
LIGHTNING_PROFILER_CHOICES = (
    None,
    LIGHTNING_PROFILER_SIMPLE,
    LIGHTNING_PROFILER_ADVANCED,
    LIGHTNING_PROFILER_PYTORCH,
)
print(f"LIGHTNING_PROFILER_CHOICES={LIGHTNING_PROFILER_CHOICES}")


LIGHTNING_GRADIENT_CLIP_ALGORITHM_VALUE = "value"  # pytorch lightning's default
LIGHTNING_GRADIENT_CLIP_ALGORITHM_NORM = "norm"
LIGHTNING_GRADIENT_CLIP_ALGORITHM_CHOICES = (
    LIGHTNING_GRADIENT_CLIP_ALGORITHM_NORM,
    LIGHTNING_GRADIENT_CLIP_ALGORITHM_VALUE, 
)


# ======================================== wandb ========================================

WANDB_WATCH_GRADIENTS = "gradients"
WANDB_WATCH_ALL = "all"
WANDB_WATCH_PARAMETERS = "parameters"
WANDB_WATCH_CHOICES = (
    None,
    WANDB_WATCH_GRADIENTS,
    WANDB_WATCH_ALL,
    WANDB_WATCH_PARAMETERS,
)
print(f"WANDB_WATCH_CHOICES={WANDB_WATCH_CHOICES}")


WANDB_CHECKPOINT_MODE_LAST = "last"
# WANDB_CHECKPOINT_MODE_BEST = "best"
# WANDB_CHECKPOINT_MODE_ALL = "all"
WANDB_CHECKPOINT_MODES = (
    None,
    WANDB_CHECKPOINT_MODE_LAST,
    # WANDB_CHECKPOINT_MODE_BEST,
    # WANDB_CHECKPOINT_MODE_ALL, 
)
print(f"WANDB_CHECKPOINT_MODES={WANDB_CHECKPOINT_MODES}")


# ==========================================================================================
# ==========================================================================================
# ======================================== parser ==========================================
# ==========================================================================================
# ==========================================================================================


def none_or_str(value: str):
    if value.lower() == 'none':
        return None
    return value


def none_or_int(value: str):
    if value.lower() == 'none':
        return None
    return int(value)


def parser_add_arguments(parser: ArgumentParser) -> ArgumentParser:
    """
    Defines all the arguments for running an FCDD experiment.
    :param parser: instance of an ArgumentParser.
    :return: the parser with added arguments
    """
    # ===================================== training =====================================
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--learning-rate', type=float)
    parser.add_argument('--weight-decay', type=float)
    # ====================================== model =======================================
    parser.add_argument('--model', type=str, choices=MODEL_CHOICES,)
    parser.add_argument("--loss", type=str, choices=LOSS_CHOICES,)
    parser.add_argument('--optimizer', type=str, choices=OPTIMIZER_CHOICES,)
    parser.add_argument(
        '--scheduler', type=none_or_str, choices=SCHEDULER_CHOICES,
        help='The type of learning rate scheduler.'
             '"lambda", reduces the learning rate each epoch by a certain factor.'
    )
    parser.add_argument(
        '--scheduler-parameters', type=float, nargs='*',
        help='Sequence of learning rate scheduler parameters. '
             '"lambda": one parameter is allowed, the factor the learning rate is reduced per epoch. '
    )
    # ====================================== dataset =====================================
    parser.add_argument('--dataset', type=str, choices=DATASET_CHOICES)
    parser.add_argument("--raw-shape", type=int, nargs=2,)
    parser.add_argument("--net-shape", type=int, nargs=2,)
    parser.add_argument('--batch-size', type=int,)
    parser.add_argument('--nworkers', type=int, help='Number of workers for data loading (DataLoader parameter).')
    parser.add_argument('--pin-memory', action='store_true')
    parser.add_argument(
        '--preprocessing', type=str, choices=ALL_PREPROCESSING_CHOICES,
        help='Preprocessing pipeline (augmentations and such). Defined inside each dataset module.'
    )
    parser.add_argument(
        '--supervise-mode', type=str, choices=ALL_SUPERVISE_MODE_CHOICES,
        help='This determines the kind of artificial anomalies. '
    )
    parser.add_argument(
        '--real-anomaly-limit', type=int,
        help='Determines the number of real anomalous images used in the case of real anomaly supervision. '
             'Has no impact on synthetic anomalies.'
    )
    # ====================================== script ======================================
    parser.add_argument(
        '--no-test', dest="test", action="store_false",
        help='If set then the model will not be tested at the end of the training. It will by default.'
    )
    parser.add_argument(
        "--preview-nimages", type=int, help="Number of images to preview per class (normal/anomalous)."
    )
    parser.add_argument(
        '--classes', type=int, nargs='+', default=None,
        help='Run only training sessions for some of the classes being nominal. If not give (default) then all classes are trained.'
    )
    parser.add_argument(
        "--cuda-visible-devices", type=int, nargs='*', default=None,
    )
    # ====================================== seeds ======================================
    seeds_group = parser.add_mutually_exclusive_group(required=True)
    seeds_group.add_argument(
        '--n-seeds', type=int, 
        help='Number of (randomly generated) seeds per class. If seeds is specified this is unnecessary.'
    )
    seeds_group.add_argument(
        "--seeds", type=seed_str2int, nargs='*',
        help="If set, the model will be trained with the given seeds."
            "Otherwise it can they can be autogenerated with -n-seeds."
            "The seeds must be passed in hexadecimal format, e.g. 0x1234."
    )
    # ====================================== files =======================================
    parser.add_argument(
        '--logdir', type=Path, default=Path("../../data/results"),
        help='Where log data is to be stored. A subfolder (rundir) is created in there. The start time is put after the dir name. Default: ../../data/results.'
    )
    parser.add_argument('--rundir-suffix', type=str, default='',)
    parser.add_argument('--rundir-prefix', type=str, default='',)
    parser.add_argument(
        '--datadir', type=Path, default=Path("../../data/datasets"),
        help='Where datasets are found or to be downloaded to. Default: ../../data/datasets.',
    )
    # ====================================== wandb =======================================
    parser.add_argument("--wandb-project", type=str,)
    parser.add_argument("--wandb-tags", type=str, nargs='*', action='extend',)
    parser.add_argument(
        "--wandb-profile", action="store_true",
        help="If set, the run will be profiled and sent to wandb."
    )
    parser.add_argument("--wandb-offline", action="store_true", help="If set, will not sync with the webserver.",)
    parser.add_argument(
        # choices taken from wandb/sdk/wandb_watch.py => watch()
        "--wandb-watch", type=none_or_str, choices=WANDB_WATCH_CHOICES, 
        help="Argument for wandb_logger.watch(..., log=WANDB_WATCH).",
    )
    parser.add_argument(
        "--wandb-watch-log-freq", type=int, default=100,
        help="Log frequency of gradients and parameters. Argument for wandb_logger.watch(..., log_freq=WANDB_WATCH_LOG_FREQ). ",
    )
    parser.add_argument(        
        "--wandb-checkpoint-mode", type=none_or_str, choices=WANDB_CHECKPOINT_MODES,
        help="How to save checkpoints."
    )
    parser.add_argument(
        "--wandb-log-roc", type=bool, nargs=3,
        help="If set, the ROC AUC curve will be logged, respectively, for train/validation/test. On test the curve is also logged."
    )
    parser.add_argument(
        "--wandb-log-pr", type=bool, nargs=3,
        help="If set, the average precision curve will be logged, respectively, for train/validation/test. On test the PR curve is also logged."
    )
    parser.add_argument(
        "--wandb-log-histogram-score", type=none_or_str, nargs=3, choices=LOG_HISTOGRAM_MODES,
        help="if and how to log  score values; you should give 3 values, respectively for the train/validation/test hooks"
    )
    parser.add_argument(
        "--wandb-log-histogram-loss", type=none_or_str, nargs=3, choices=LOG_HISTOGRAM_MODES,
        help="if and how to log loss values; you should give 3 values, respectively for the train/validation/test hooks"
    )
    group_log_image_heatmap = parser.add_argument_group("log-image-heatmap")
    group_log_image_heatmap.add_argument(
        "--wandb-log-image-heatmap-contrast-percentiles", type=float, nargs=2,
        help="Percentile values for the contrast of the heatmap: min/max"
    )
    group_log_image_heatmap.add_argument(
        "--wandb-log-image-heatmap-nsamples", nargs=3, type=int,
        help="how many of each class (normal/anomalous) per epoch?"
             "alwyas log the same ones assuming the order of images doesnt' change"
    )
    group_log_image_heatmap.add_argument(
        "--wandb-log-image-heatmap-resolution", nargs=3, type=none_or_int,
        help="size of the image (width=height), and if 'None' then keep the original size"
    )
    parser.add_argument(
        "--wandb-log-percentiles-score-train", type=float, nargs="*",
        help="If set, the score will be logged at the given percentiles for train (normal and anomalous scores separatedly)."
    )
    parser.add_argument(
        "--wandb-log-percentiles-score-validation", type=float, nargs="*",
        help="If set, the score will be logged at the given percentiles for train (normal and anomalous scores separatedly)."
    )
    parser.add_argument(
        "--wandb-log-perinstance-mean-score", type=bool, nargs=3,
        help="if true then the pixel score will be averaged per instance and logged in a table"
    )
    parser.add_argument(
        "--wandb-log-perinstance-mean-loss", type=bool, nargs=3,
        help="if true then the loss will be averaged per instance and logged in a table; you should give 3 values, respectively for the train/validation/test hooks"
    )
    
    # ================================ pytorch lightning =================================
    parser.add_argument(
        "--lightning-accelerator", type=str, 
        default=LIGHTNING_ACCELERATOR_GPU, 
        choices=LIGHTNING_ACCELERATOR_CHOICES,
        help=f"https://pytorch-lightning.readthedocs.io/en/latest/extensions/accelerator.html"
    )
    parser.add_argument("--lightning-ndevices", type=int, default=1, help="Number of devices (gpus) to use for training.")
    parser.add_argument(
        "--lightning-strategy", type=none_or_str, 
        default=None, 
        choices=LIGHTNING_STRATEGY_CHOICES,
        help="See https://pytorch-lightning.readthedocs.io/en/latest/extensions/strategy.html"
    )
    parser.add_argument(
        "--lightning-precision", type=int, choices=LIGHTNING_PRECISION_CHOICES,
        help="https://pytorch-lightning.readthedocs.io/en/latest/guides/speed.html#mixed-precision-16-bit-training"
    )
    parser.add_argument(
        "--lightning-model-summary-max-depth", type=int, 
        help="https://pytorch-lightning.readthedocs.io/en/latest/api/pytorch_lightning.callbacks.ModelSummary.html#pytorch_lightning.callbacks.ModelSummary",
    )
    parser.add_argument(
        "--lightning-check-val-every-n-epoch", type=int,
        help="... find link to lightning doc ..."
    )
    parser.add_argument(
        "--lightning-accumulate-grad-batches", type=int,
        help="Accumulate gradients for THIS batches. https://pytorch-lightning.readthedocs.io/en/latest/advanced/training_tricks.html#accumulate-gradients"
    )
    parser.add_argument(
        "--lightning-profiler", type=none_or_str, choices=LIGHTNING_PROFILER_CHOICES,
        help="simple and advanced: https://pytorch-lightning.readthedocs.io/en/latest/tuning/profiler_basic.html\n"
             "pytorch: https://pytorch-lightning.readthedocs.io/en/latest/tuning/profiler_intermediate.html\n"
             "in any case it is saved in a f"
    )
    group_gradient_clipping = parser.add_argument_group("gradient-clipping")
    group_gradient_clipping.add_argument(
        "--lightning-gradient-clip-val", type=float,
        help=f"https://pytorch-lightning.readthedocs.io/en/latest/advanced/training_tricks.html#gradient-clipping",
    )
    group_gradient_clipping.add_argument(
        "--lightning-gradient-clip-algorithm", type=str, choices=LIGHTNING_GRADIENT_CLIP_ALGORITHM_CHOICES,
        help=f"https://pytorch-lightning.readthedocs.io/en/latest/advanced/training_tricks.html#gradient-clipping",
    )
    parser.add_argument(
        "--lightning-deterministic", type=bool,
        help="https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html#reproducibility",
    )
    return parser


def args_validate_dataset_specific_choices(args):
    assert args.preprocessing in dataset_preprocessing_choices(args.dataset), f"args.dataset={args.dataset}: {args.preprocessing} not in {dataset_preprocessing_choices(args.dataset)}"
    assert args.supervise_mode in dataset_supervise_mode_choices(args.dataset), f"args.dataset={args.dataset}: {args.supervise_mode} not in {dataset_supervise_mode_choices(args.dataset)}"


def args_validate_model_specific_choices(args):
    assert args.loss in model_loss_choices(args.model), f"args.model={args.model}: {args.loss} not in {model_loss_choices(args.model)}"
    assert args.optimizer in model_optimizer_choices(args.model), f"args.model={args.model}: {args.optimizer} not in {model_optimizer_choices(args.model)}"
    assert args.scheduler in model_scheduler_choices(args.model), f"args.model={args.model}: {args.scheduler} not in {model_scheduler_choices(args.model)}"


def args_post_parse(args_):
    
    # ================================== start time ==================================
    def time_format(i: float) -> str:
        """ takes a timestamp (seconds since epoch) and transforms that into a datetime string representation """
        return datetime.fromtimestamp(i).strftime('%Y%m%d%H%M%S')
    args_.script_start_time = int(time.time())

    # ================================== list 2 tuple (imutable) ==================================
    args_.raw_shape = tuple(args_.raw_shape)
    args_.net_shape = tuple(args_.net_shape)
    
    # ================================== cuda ==================================
    
    if args_.cuda_visible_devices is not None:
        args_.cuda_visible_devices = tuple(args_.cuda_visible_devices)
        
    # ================================== paths ==================================
    args_.datadir = args_.datadir.resolve().absolute()
    
    args_.rundir_prefix += '_' if args_.rundir_prefix else ''
    args_.rundir_suffix = f"_{args_.rundir_suffix}" if args_.rundir_suffix else ''
    rundir_name = f"{args_.rundir_prefix}run_{args_.script_start_time}{args_.rundir_suffix}"
    del vars(args_)['rundir_suffix']
    del vars(args_)['rundir_prefix']
    
    logdir = args_.logdir.resolve().absolute()
    if args_.wandb_project is not None:
        logdir = logdir / args_.wandb_project    
    logdir = logdir / args_.dataset
    args_.logdir = logdir / rundir_name
    
    # ================================== seeds ==================================
    
    if args_.seeds is not None:
        for s in args_.seeds:
            assert type(s) == int, f"seed must be an int, got {type(s)}"
            assert s >= 0, f"seed must be >= 0, got {s}"
        assert len(set(args_.seeds)) == len(args_.seeds), f"seeds must be unique, got {s}"
        args_.seeds = tuple(args_.seeds)
        
    # seeds and n_seeds are in a mutually exclusive group in the parser
    # so the case below must have n_seeds 
    else:
        seeds = []
        for _ in range(args_.n_seeds):
            seeds.append(create_seed())
            time.sleep(1/3)  # let the system state change
        args_.seeds = tuple(seeds)
        
    del vars(args_)['n_seeds']
    
    # ================================== wandb_log_percentiles_score ==================================
        
    args_.wandb_log_percentiles_score = (
        args_.wandb_log_percentiles_score_train,
        args_.wandb_log_percentiles_score_validation,
    ) 
    del vars(args_)['wandb_log_percentiles_score_train']
    del vars(args_)['wandb_log_percentiles_score_validation']
        
    return args_


# ==========================================================================================
# ==========================================================================================
# ========================================== run one =======================================
# ==========================================================================================
# ==========================================================================================


def run_one(
    # training
    epochs: int,
    learning_rate: float, 
    weight_decay: float, 
    # model
    model: str, 
    loss: str,
    optimizer: str, 
    scheduler: str,
    scheduler_parameters: list,
    # dataset
    dataset: str,
    raw_shape: Tuple[int, int],
    net_shape: Tuple[int, int],
    batch_size: int,
    nworkers: int, 
    pin_memory: bool,
    preprocessing: str,
    supervise_mode: str,
    real_anomaly_limit: int,
    # dataset (run-specific)
    normal_class: int,
    # script
    test: bool,
    preview_nimages: int,
    # script (run-specific)
    seed: int,
    # files
    logdir: Path,
    datadir: Path,
    # wandb
    wandb_logger: WandbLogger,
    wandb_profile: bool,
    wandb_watch: Optional[str],
    wandb_watch_log_freq: int,
    # wandb (train/validation/test)
    # each one below is a tuple of 3 things
    # which respectively configure the train/validation/test phases
    wandb_log_roc: Tuple[bool, bool, bool],
    wandb_log_pr: Tuple[bool, bool, bool],
    wandb_log_image_heatmap_contrast_percentiles: Tuple[float, float],
    wandb_log_image_heatmap_nsamples: Tuple[int, int, int],
    wandb_log_image_heatmap_resolution: Tuple[int, int, int],
    wandb_log_histogram_score: Tuple[bool, bool, bool],
    wandb_log_histogram_loss: Tuple[bool, bool, bool],
    wandb_log_percentiles_score: Tuple[Tuple[float, ...], Tuple[float, ...]],
    wandb_log_perinstance_mean_score: Tuple[bool, bool, bool],
    wandb_log_perinstance_mean_loss: Tuple[bool, bool, bool], 
    # pytorch lightning
    lightning_accelerator: str,
    lightning_ndevices: int,
    lightning_strategy: str,
    lightning_precision: str,
    lightning_model_summary_max_depth: int,
    lightning_check_val_every_n_epoch: int,
    lightning_accumulate_grad_batches: int,
    lightning_profiler: str, 
    lightning_gradient_clip_val: float,
    lightning_gradient_clip_algorithm: str,
    lightning_deterministic: bool, 
):
    
    # minimal validation for early mistakes
    assert dataset in DATASET_CHOICES, f"Invalid dataset: {dataset}, chose from {DATASET_CHOICES}"
    assert supervise_mode in dataset_supervise_mode_choices(dataset), f"Invalid supervise_mode: {supervise_mode} for dataset {dataset}, chose from {dataset_supervise_mode_choices(dataset)}"
    assert preprocessing in dataset_preprocessing_choices(dataset), f"Invalid preproc: {preprocessing} for dataset {dataset}, chose from {dataset_preprocessing_choices(dataset)}"
    assert loss in model_loss_choices(model), f"Invalid loss: {loss} for model {model}, chose from {model_loss_choices(model)}"
    assert optimizer in model_optimizer_choices(model), f"Invalid optimizer: {optimizer} for model {model}, chose from {model_optimizer_choices(model)}"
    assert scheduler in model_scheduler_choices(model), f"Invalid scheduler: {scheduler} for model {model}, chose from {model_scheduler_choices(model)}"
    assert lightning_accelerator in LIGHTNING_ACCELERATOR_CHOICES, f"Invalid lightning_accelerator: {lightning_accelerator}, chose from {LIGHTNING_ACCELERATOR_CHOICES}"
    if lightning_strategy is not None:
        assert lightning_strategy in LIGHTNING_STRATEGY_CHOICES, f"Invalid lightning_strategy: {lightning_strategy}, chose from {LIGHTNING_STRATEGY_CHOICES}"
    assert lightning_precision in LIGHTNING_PRECISION_CHOICES, f"Invalid lightning_precision: {lightning_precision}, chose from {LIGHTNING_PRECISION_CHOICES}"
    assert lightning_profiler in LIGHTNING_PROFILER_CHOICES, f"Invalid lightning_profiler: {lightning_profiler}, chose from {LIGHTNING_PROFILER_CHOICES}"
    
    assert len(wandb_log_roc) == 3, f"wandb_log_roc should have 3 bools, for train/validation/test, but got {wandb_log_roc}" 
    assert all(isinstance(obj, bool) for obj in wandb_log_roc), f"wandb_log_roc should only have bool, got {wandb_log_roc}"
    assert len(wandb_log_pr) == 3, f"wandb_log_pr should have 3 bools, for train/validation/test, but got {wandb_log_pr}" 
    assert all(isinstance(obj, bool) for obj in wandb_log_pr), f"wandb_log_pr should only have bool, got {wandb_log_pr}"
    
    assert len(wandb_log_histogram_score) == 3, f"wandb_log_histogram_score should have 3 bools, for train/validation/test, but got {len(wandb_log_histogram_score)} things"
    assert all(val in LOG_HISTOGRAM_MODES for val in wandb_log_histogram_score), f"wandb_log_histogram_score values should be in {LOG_HISTOGRAM_MODES}, got {wandb_log_histogram_score}"
    assert len(wandb_log_histogram_loss) == 3, f"wandb_log_histogram_loss should have 3 bools, for train/validation/test, but got {len(wandb_log_histogram_loss)} things"
    assert all(val in LOG_HISTOGRAM_MODES for val in wandb_log_histogram_loss), f"wandb_log_histogram_loss values should be in {LOG_HISTOGRAM_MODES}, got {wandb_log_histogram_loss}"

    logdir.mkdir(parents=True, exist_ok=True)
    
    # seed
    (logdir / "seed.txt").write_text(seed_int2str(seed))
    torch.manual_seed(seed)
    
    batch_size_effective = batch_size * lightning_accumulate_grad_batches if lightning_accumulate_grad_batches > 0 else batch_size
    
    summary_update_dict_inputs = dict(
        batch_size_effective=batch_size_effective,
    )
    wandb.run.summary.update(summary_update_dict_inputs)
    
    # ================================ DATA ================================
    
    # datamodule hard-coded for now, but later other datasets can be added
    datamodule = mvtec_dataset_dev01.MVTecAnomalyDetectionDataModule(
        root=datadir,
        normal_class=normal_class,
        preprocessing=preprocessing,
        supervise_mode=supervise_mode,
        real_anomaly_limit=real_anomaly_limit,
        raw_shape=raw_shape,
        net_shape=net_shape,
        batch_size=batch_size,
        nworkers=nworkers,
        pin_memory=pin_memory,
        seed=seed,
    )
    summary_update_dict_dataset = dict(
        normal_class_label=dataset_class_labels(dataset)[normal_class],
        mvtec_class_type=mvtec_dataset_dev01.CLASSES_TYPES[normal_class],
        normal_class_fullqualified=dataset_class_fullqualified(dataset)[normal_class],
    )
    wandb.run.summary.update(summary_update_dict_dataset)
    datamodule.prepare_data()

    # ================================ MODEL ================================
    try:
        model_class = MODEL_CLASSES[model]
       
    except KeyError as err:
        raise NotImplementedError(f'Model {model} is not implemented!') from err

    try:
        model = model_class(
            in_shape=datamodule.net_shape, 
            loss_name=loss,
            # optimizer
            optimizer_name=optimizer,
            lr=learning_rate,
            weight_decay=weight_decay,
            # scheduler
            scheduler_name=scheduler,
            scheduler_parameters=scheduler_parameters,
        )
        
    except ModelError as ex:
        msg = ex.args[0]
        if "required positional arguments" in msg:
            raise ModelError(f"Model model_class={model_class.__name__} requires positional argument missing ") from ex
        
    def test_input_output_dimensions(inshape):    
        random_tensor = torch.ones((1, 3, inshape[0], inshape[1]))
        scores = model(random_tensor)
        assert scores.shape[-2:] == inshape, f"{model.__class__.__name__} must return a tensor of shape (..., {inshape[0]}, {inshape[1]}), found {scores.shape}"
        assert tuple(scores.shape[0:2]) == (1, 1), f"{model.__class__.__name__} must return a tensor of shape (1, 1, ...), found {scores.shape}"
    
    test_input_output_dimensions(datamodule.net_shape)
    
    def log_model_architecture(model_: torch.nn.Module):
        model_str = str(model_)
        model_str_fpath = logdir / "model.txt"
        model_str_fpath.write_text(model_str)
         # now = dont keep syncing if it changes
        wandb.save(str(model_str_fpath), policy="now") 
    
    log_model_architecture(model)
    
    if wandb_watch is not None:
        wandb_logger.watch(model, log=wandb_watch, log_freq=wandb_watch_log_freq)

    # ================================ CALLBACKS ================================

    callbacks = [
        # pl.callbacks.ModelSummary(max_depth=lightning_model_summary_max_depth),
        pl.callbacks.RichModelSummary(max_depth=lightning_model_summary_max_depth),
        LearningRateLoggerCallback(),
    ]
    
    # ========================================================================= ROC

    def add_callbacks_log_roc(train: bool, validation: bool, test: bool):
    
        if train:
            callbacks.append(
                LogRocCallback(
                    stage=RunningStage.TRAINING,
                    scores_key="score_maps",
                    gt_key="gtmaps",
                    log_curve=False,
                    limit_points=PIXELWISE_ROC_LIMIT_POINTS,
                    python_generator=create_python_random_generator(seed),
                )
            )
            
        if validation:
            callbacks.append(
                LogRocCallback(
                    stage=RunningStage.VALIDATING,
                    scores_key="score_maps",
                    gt_key="gtmaps",
                    log_curve=False,
                    limit_points=PIXELWISE_ROC_LIMIT_POINTS, 
                    python_generator=create_python_random_generator(seed),
                )
            )
            
        if test:
            callbacks.append(
                LogRocCallback(
                    stage=RunningStage.TESTING,
                    scores_key="score_maps",
                    gt_key="gtmaps",
                    log_curve=True,
                    limit_points=None,
                    python_generator=create_python_random_generator(seed),
                )
            )
            
    add_callbacks_log_roc(*wandb_log_roc)
    
    # ========================================================================= PR
    
    def add_callbacks_log_pr(train: bool, validation: bool, test: bool):
    
        if train:
            callbacks.append(
                LogAveragePrecisionCallback(
                    stage=RunningStage.TRAINING,
                    scores_key="score_maps",
                    gt_key="gtmaps",
                    log_curve=False,
                    limit_points=PIXELWISE_PR_LIMIT_POINTS, 
                    python_generator=create_python_random_generator(seed),
                )
            )
        if validation:
            callbacks.append(
                LogAveragePrecisionCallback(
                    stage=RunningStage.VALIDATING,
                    scores_key="score_maps",
                    gt_key="gtmaps",
                    log_curve=False,
                    limit_points=PIXELWISE_PR_LIMIT_POINTS,  
                    python_generator=create_python_random_generator(seed),
                )
            )
        if test:
            callbacks.append(
                LogAveragePrecisionCallback(
                    stage=RunningStage.TESTING,
                    scores_key="score_maps",
                    gt_key="gtmaps",
                    log_curve=True,
                    limit_points=None,
                    python_generator=create_python_random_generator(seed),
                )
            )
            
    add_callbacks_log_pr(*wandb_log_pr)        
    
    # ========================================================================= heatmaps
    
    def add_callbacks_log_image_heatmap(
        nsamples_train: int, nsamples_validation: int, nsamples_test: int, 
        resolution_train: Optional[int], resolution_validation: Optional[int], resolution_test: Optional[int],
    ):
    
        if nsamples_train > 0:
            callbacks.extend([     
                LogImageHeatmapTableCallback(
                    stage=RunningStage.TRAINING,
                    imgs_key="inputs",
                    scores_key="score_maps",
                    masks_key="gtmaps",
                    labels_key="labels",
                    nsamples_each_class=nsamples_train,
                    resolution=resolution_train,
                    heatmap_normalization=HEATMAP_NORMALIZATION_PERCENTILES_IN_EPOCH,
                    min_max_percentiles=wandb_log_image_heatmap_contrast_percentiles,
                    python_generator=create_python_random_generator(seed),
                ),
            ])
        
        if nsamples_validation > 0:
            callbacks.extend([     
                LogImageHeatmapTableCallback(
                    stage=RunningStage.VALIDATING,
                    imgs_key="inputs",
                    scores_key="score_maps",
                    masks_key="gtmaps",
                    labels_key="labels",
                    nsamples_each_class=nsamples_validation,
                    resolution=resolution_validation,
                    heatmap_normalization=HEATMAP_NORMALIZATION_PERCENTILES_IN_EPOCH,
                    min_max_percentiles=wandb_log_image_heatmap_contrast_percentiles,
                    python_generator=create_python_random_generator(seed),
                ),
            ])
        
        if nsamples_test > 0:
            callbacks.extend([     
                LogImageHeatmapTableCallback(
                    stage=RunningStage.TESTING,
                    imgs_key="inputs",
                    scores_key="score_maps",
                    masks_key="gtmaps",
                    labels_key="labels",
                    nsamples_each_class=nsamples_test,
                    resolution=resolution_test,
                    heatmap_normalization=HEATMAP_NORMALIZATION_PERCENTILES_IN_EPOCH,
                    min_max_percentiles=wandb_log_image_heatmap_contrast_percentiles,
                    python_generator=create_python_random_generator(seed),
                ),
            ])
    
    add_callbacks_log_image_heatmap(
        *wandb_log_image_heatmap_nsamples, *wandb_log_image_heatmap_resolution,
    )
    # ========================================================================= histograms
    
    def add_callbacks_log_histogram_score(train_mode: str, validation_mode: str, test_mode: str):
        
        if train_mode is not None:
            callbacks.extend([
                LogHistogramCallback(stage=RunningStage.TRAINING, key="score_maps", mode=train_mode,), 
                LogHistogramsSuperposedPerClassCallback(
                    stage=RunningStage.TRAINING, 
                    mode=train_mode,
                    limit_points=PIXELWISE_HISTOGRAMS_LIMIT_POINTS_TRAIN,  # 3k
                    # same everywhere 
                    values_key="score_maps", 
                    gt_key="gtmaps", 
                    python_generator=create_python_random_generator(seed),
                ),
            ])

        if validation_mode is not None:
            callbacks.extend([
                LogHistogramCallback(stage=RunningStage.VALIDATING, key="score_maps", mode=validation_mode,), 
                LogHistogramsSuperposedPerClassCallback(
                    stage=RunningStage.VALIDATING, 
                    mode=validation_mode,
                    limit_points=PIXELWISE_HISTOGRAMS_LIMIT_POINTS_VALIDATION,  # 3k
                    # same everywhere 
                    values_key="score_maps", 
                    gt_key="gtmaps", 
                    python_generator=create_python_random_generator(seed),
                ),
            ])
        
        if test_mode is not None:
            callbacks.extend([
                LogHistogramCallback(stage=RunningStage.TESTING, key="score_maps", mode=test_mode,), 
                LogHistogramsSuperposedPerClassCallback(
                    stage=RunningStage.TESTING, 
                    mode=test_mode,
                    limit_points=PIXELWISE_HISTOGRAMS_LIMIT_POINTS_TEST,  # 30k
                    # same everywhere 
                    values_key="score_maps", 
                    gt_key="gtmaps", 
                    python_generator=create_python_random_generator(seed),
                ),
            ])
            
    add_callbacks_log_histogram_score(*wandb_log_histogram_score)
    
    
    def add_callbacks_log_histogram_loss(train_mode: str, validation_mode: str, test_mode: str):
        
        if train_mode is not None:
            callbacks.extend([
                LogHistogramCallback(stage=RunningStage.TRAINING, key="loss_maps", mode=train_mode,), 
                LogHistogramsSuperposedPerClassCallback(
                    stage=RunningStage.TRAINING, 
                    mode=train_mode,
                    limit_points=PIXELWISE_HISTOGRAMS_LIMIT_POINTS_TRAIN,  # 3k
                    # same everywhere 
                    values_key="loss_maps", 
                    gt_key="gtmaps", 
                    python_generator=create_python_random_generator(seed),
                ),
            ])

        if validation_mode is not None:
            callbacks.extend([
                LogHistogramCallback(stage=RunningStage.VALIDATING, key="loss_maps", mode=validation_mode,), 
                LogHistogramsSuperposedPerClassCallback(
                    stage=RunningStage.VALIDATING, 
                    mode=validation_mode,
                    limit_points=PIXELWISE_HISTOGRAMS_LIMIT_POINTS_VALIDATION,  # 3k
                    # same everywhere 
                    values_key="loss_maps", 
                    gt_key="gtmaps", 
                    python_generator=create_python_random_generator(seed),
                ),
            ])
        
        if test_mode is not None:
            callbacks.extend([
                LogHistogramCallback(stage=RunningStage.TESTING, key="loss_maps", mode=test_mode,), 
                LogHistogramsSuperposedPerClassCallback(
                    stage=RunningStage.TESTING, 
                    mode=test_mode,
                    limit_points=PIXELWISE_HISTOGRAMS_LIMIT_POINTS_TEST,  # 30k
                    # same everywhere 
                    values_key="loss_maps", 
                    gt_key="gtmaps", 
                    python_generator=create_python_random_generator(seed),
                ),
            ])
        
    add_callbacks_log_histogram_loss(*wandb_log_histogram_loss)

    if preview_nimages > 0:
        datamodule.setup("fit")
        callbacks.append(
            DataloaderPreviewCallback(
                dataloader=datamodule.train_dataloader(embed_preprocessing=True), 
                n_samples=preview_nimages, logkey_prefix="train/preview",
            )
        )
    
    # ================================ PERCENTILES ================================
    
    def add_callbacks_log_percentiles_score(train: Tuple[float, ...], validation: Tuple[float, ...]):
    
        if len(train) > 0:
            callbacks.append(
                LogPercentilesPerClassCallback(
                    stage=RunningStage.TRAINING, 
                    values_key="score_maps",
                    gt_key="gtmaps", 
                    percentiles=train,
                )
            )
                     
        if len(validation) > 0:
            callbacks.append(
                LogPercentilesPerClassCallback(
                    stage=RunningStage.VALIDATING, 
                    values_key="score_maps",
                    gt_key="gtmaps", 
                    percentiles=validation,
                )
            )           
    
    add_callbacks_log_percentiles_score(*wandb_log_percentiles_score)
    
    # ================================ PER-INSTANCE ================================
    
    # score
    def add_callbacks_log_perinstance_mean_score(train: bool, validation: bool, test: bool):
    
        if train:
            callbacks.append(
                LogPerInstanceValueCallback(
                    stage=RunningStage.TRAINING,
                    values_key="score_maps",
                    labels_key="labels",
                )
            )
                     
        if validation:
            callbacks.append(
                LogPerInstanceValueCallback(
                    stage=RunningStage.VALIDATING,
                    values_key="score_maps",
                    labels_key="labels",
                )
            )    
        
        if test:
            callbacks.append(
                LogPerInstanceValueCallback(
                    stage=RunningStage.TESTING,
                    values_key="score_maps",
                    labels_key="labels",
                )
            )
    # loss            
    def add_callbacks_log_perinstance_mean_loss(train: bool, validation: bool, test: bool):
    
        if train:
            callbacks.append(
                LogPerInstanceValueCallback(
                    stage=RunningStage.TRAINING,
                    values_key="loss_maps",
                    labels_key="labels",
                )
            )
                     
        if validation:
            callbacks.append(
                LogPerInstanceValueCallback(
                    stage=RunningStage.VALIDATING,
                    values_key="loss_maps",
                    labels_key="labels",
                )
            )    
        
        if test:
            callbacks.append(
                LogPerInstanceValueCallback(
                    stage=RunningStage.TESTING,
                    values_key="loss_maps",
                    labels_key="labels",
                )
            )
               
    add_callbacks_log_perinstance_mean_score(*wandb_log_perinstance_mean_score)
    add_callbacks_log_perinstance_mean_loss(*wandb_log_perinstance_mean_loss)

    # ================================ PROFILING ================================
    
    if not wandb_profile:
        profiler = contextlib.nullcontext()
    
    else:
        # Set up profiler
        wait, warmup, active, repeat = 1, 1, 2, 1
        schedule =  torch.profiler.schedule(wait=wait, warmup=warmup, active=active, repeat=repeat,)
        profile_dir = Path(f"{wandb_logger.save_dir}/latest-run/tbprofile").absolute()
        profiler = torch.profiler.profile(
            schedule=schedule, 
            on_trace_ready=tensorboard_trace_handler(str(profile_dir)), 
            with_stack=True,
        )
        callbacks.append(TorchTensorboardProfilerCallback(profiler))
    
    def get_lightning_profiler(profiler_choice):
        
        if profiler_choice is None:
            return None
        
        elif profiler_choice == LIGHTNING_PROFILER_SIMPLE:
            return SimpleProfiler(
                dirpath=wandb_logger.save_dir,
                filename=f"lightning-profiler.{LIGHTNING_PROFILER_SIMPLE}",
                extended=True,
            )
        
        elif profiler_choice == LIGHTNING_PROFILER_ADVANCED:
            return AdvancedProfiler(
                dirpath=wandb_logger.save_dir,
                filename=f"lightning-profiler.{LIGHTNING_PROFILER_ADVANCED}",

            )
        
        elif profiler_choice == LIGHTNING_PROFILER_PYTORCH:
            return PyTorchProfiler(
                dirpath=wandb_logger.save_dir,
                filename=f"lightning-profiler.{LIGHTNING_PROFILER_PYTORCH}",
                sort_by_key="cuda_time_total",
            )
        
        else:
            raise NotImplementedError(f"Profiler {profiler_choice} not implemented.")

    trainer_profiler = get_lightning_profiler(lightning_profiler)
    
    # ================================ FIT ================================
    trainer = pl.Trainer(
        accelerator=lightning_accelerator,
        gpus=lightning_ndevices, 
        strategy=lightning_strategy,
        precision=lightning_precision,
        logger=wandb_logger,  
        log_every_n_steps=1,  
        max_epochs=epochs,    
        callbacks=callbacks, 
        # i should learn how to properly deal with
        # the sanity check but for now it's causing too much trouble
        num_sanity_val_steps=0,
        check_val_every_n_epoch=lightning_check_val_every_n_epoch,
        accumulate_grad_batches=lightning_accumulate_grad_batches,
        profiler=trainer_profiler,
        gradient_clip_val=lightning_gradient_clip_val,
        gradient_clip_algorithm=lightning_gradient_clip_algorithm,
        deterministic=lightning_deterministic,
    )
    
    with profiler:
        # datamodule.setup("fit")
        trainer.fit(model=model, datamodule=datamodule)
    
    if wandb_watch is not None:
        wandb_logger.experiment.unwatch(model)

    if wandb_profile:
        profile_art = wandb.Artifact(f"trace-{wandb.run.id}", type="profile")
        profile_art.add_file(str(next(profile_dir.glob("*.pt.trace.json"))), "trace.pt.trace.json")
        wandb_logger.experiment.log_artifact(profile_art)
    
    if trainer_profiler is not None:
        wandb.save(str(Path(trainer_profiler.dirpath) / trainer_profiler.filename), policy="now") 

    if not test:
        print(f"experiment folder: {Path(wandb_logger.save_dir).resolve().absolute()}")
        return 
    
    # ================================ TEST ================================
    trainer.test(model=model, datamodule=datamodule)    
    print(f"experiment folder: {Path(wandb_logger.save_dir).resolve().absolute()}")
        
# ==========================================================================================
# ==========================================================================================
# ========================================== run ===========================================
# ==========================================================================================
# ==========================================================================================

    
def run(**kwargs) -> dict:
    """see the arguments in run_one()"""
    
    # todo:refactor: make the cuda device selection a decorator
    cuda_visible_devices = kwargs.pop("cuda_visible_devices", None)
    if cuda_visible_devices is not None:
        print(f"Using cuda devices: {cuda_visible_devices} ==> setting environment variable CUDA_VISIBLE_DEVICES")
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, cuda_visible_devices))
    
    script_start_time = kwargs.pop("script_start_time")
    
    base_logdir = kwargs['logdir']
    dataset = kwargs['dataset']
    
    base_logdir = base_logdir / dataset
    
    wandb_offline = kwargs.pop("wandb_offline", False)
    wandb_project = kwargs.pop("wandb_project", None)
    wandb_tags = kwargs.pop("wandb_tags", None) or []
    
    wandb_checkpoint_mode = kwargs.pop("wandb_checkpoint_mode")
    
    # later the case of multiple saving modes will be handled
    # for now the 'else' is just True, which means 'last'
    log_model = False if wandb_checkpoint_mode is None else True
        
    if log_model:
        assert not wandb_offline, f"wandb_offline={wandb_offline} is incompatible with log_model={log_model} (from wandb_checkpoint_mode={wandb_checkpoint_mode})"

    if wandb_offline:
        print(f"wandb_offline={wandb_offline} --> setting enviroment variable WANDB_MODE=offline")
        os.environ["WANDB_MODE"] = "offline"
    
    # if none then do all the classes
    classes = kwargs.pop("classes", None) or tuple(range(dataset_nclasses(dataset)))

    seeds = kwargs.pop('seeds')   
    its = tuple(range(len(seeds)))
    
    for c in classes:
        
        print(f"class {c:02d}")
        cls_logdir = base_logdir / f'normal_{c}'
        kwargs.update(dict(normal_class=c,))
        
        for it, seed in zip(its, seeds):
            
            print(f"it {it:02d} seed {seed_int2str(seed)}")    
            kwargs.update(dict(seed=seed,))
            
            logdir = (cls_logdir / f'it_{it:02}').absolute()
            # it's super important that the dir must already exist for wandb logging           
            logdir.mkdir(parents=True, exist_ok=True)
            print(f"logdir: {logdir}")
            
            wandb_name = f"{dataset}.{base_logdir.name}.cls{c:02}.it{it:02}"
            print(f"wandb_name={wandb_name}")
            
            run_one_kwargs = {
                **kwargs, 
                # overwrite logdir
                **dict(logdir=logdir,  seed=seed,)
            }
            
            # the ones added here don't go to the run_one()            
            wandb_config = {
                **run_one_kwargs,
                **dict(
                    seeds_str=seed_int2str(seed),
                    script_start_time=script_start_time,
                    it=it,
                    cuda_visible_devices=cuda_visible_devices,
                    script_version=__version__,
                    pid=os.getpid(),
                    slurm_cluster_name=os.environ.get("SLURM_CLUSTER_NAME"),
                    slurmd_nodename=os.environ.get("SLURMD_NODENAME"),
                    slurm_job_partition=os.environ.get("SLURM_JOB_PARTITION"),
                    slurm_submit_host=os.environ.get("SLURM_SUBMIT_HOST"),
                    slurm_job_user=os.environ.get("SLURM_JOB_USER"),
                    slurm_task_pid=os.environ.get("SLURM_TASK_PID"),
                    slurm_job_name=os.environ.get("SLURM_JOB_NAME"),
                    slurm_array_job_id=os.environ.get("SLURM_ARRAY_JOB_ID"),
                    slurm_array_task_id=os.environ.get("SLURM_ARRAY_TASK_ID"),
                    slurm_job_id=os.environ.get("SLURM_JOB_ID"),
                    slurm_job_gpus=os.environ.get("SLURM_JOB_GPUS"),
                ),
            }
            
            # add a few hashes to the make it 
            # easier to group things in wandb
            wandb_config = {
                **wandb_config,
                **dict(
                    confighash_full=hashify_config(wandb_config, keys=[
                        "epochs", "learning_rate", "weight_decay", "model", "loss", "optimizer", "scheduler", "scheduler_parameters", "dataset", "raw_shape", "net_shape", "batch_size", "preprocessing", "supervise_mode", "real_anomaly_limit", "normal_class",                         
                    ]),
                    confighash_dataset_class_supervise=hashify_config(
                        wandb_config, keys=("datset", "normal_class", "supervise_mode")
                    ),
                    confighash_dataset_class_supervise_loss=hashify_config(
                        wandb_config, keys=("datset", "normal_class", "supervise_mode", "loss")
                    ),
                    confighash_dataset_class_supervise_model=hashify_config(
                        wandb_config, keys=("datset", "normal_class", "supervise_mode", "model")
                    ),
                    confighash_dataset_class_supervise_loss_model=hashify_config(
                        wandb_config, keys=("datset", "normal_class", "supervise_mode", "loss", "model")
                    ),
                    confighash_slurm=hashify_config(
                        # the info here should be very redundant but it's ok
                        wandb_config, keys=(
                            "slurm_job_id", "slurm_array_job_id", "slurm_array_task_id", "slurm_task_pid", "slurm_job_user", "slurm_job_name", "slurm_submit_host", "slurm_cluster_name", "slurmd_nodename", "slurm_job_partition",
                        )
                    ),
                )
            }
            
            wandb_init_kwargs = dict(
                project=wandb_project, 
                name=wandb_name,
                entity="mines-paristech-cmm",
                tags=wandb_tags,
                config=wandb_config,
                save_code=True,
                reinit=True,
            )
            print(f"wandb_init_kwargs={wandb_init_kwargs}")
            wandb_logger = WandbLogger(
                save_dir=str(logdir),
                offline=wandb_offline,
                # for now only the last checkpoint is available, but later the others can be integrated
                # (more stuff have to be done in the run_one())
                log_model=log_model,
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
            
            print(f"run_one_kwargs={run_one_kwargs}")
            try:
                run_one(wandb_logger=wandb_logger, **run_one_kwargs,)
            
            except TypeError as ex:
                msg = ex.args[0]
                if "run_one() got an unexpected keyword argument" in msg:
                    raise ScriptError(f"run_one() got an unexpected keyword argument: {msg}, did you forget to kwargs.pop() something?") from ex
                raise ex
            except Exception as ex:
                # wandb_logger.finalize("failed")
                wandb.finish(1)
                raise ex
            else:
                # wandb_logger.finalize("success")
                wandb.finish(0)
                