#!/usr/bin/env python
# coding: utf-8

import contextlib
import functools
from gc import callbacks
import os
import time
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from re import A
from typing import Any, Callable, List, Optional, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.trainer.states import RunningStage
from scipy.interpolate import interp1d
from sklearn.metrics import average_precision_score, precision_recall_curve
from torch.profiler import tensorboard_trace_handler

import mvtec_dataset_dev01 as mvtec_dataset_dev01
import wandb
from callbacks_dev01 import LogAveragePrecisionCallback, LogHistogramCallback, LogHistogramsSuperposedPerClassCallback, LogRocCallback, TorchTensorboardProfilerCallback, LOG_HISTOGRAM_MODES, LOG_HISTOGRAM_MODE_NONE
from common_dev01 import (create_python_random_generator, create_seed,
                          seed_int2str, seed_str2int)
from data_dev01 import ANOMALY_TARGET, NOMINAL_TARGET

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
    model_dev01.FCDD_CNN224_VGG.__name__: model_dev01.FCDD_CNN224_VGG,
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
        model_dev01.FCDD_CNN224_VGG.__name__: model_dev01.OPTIMIZER_CHOICES,
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
        model_dev01.FCDD_CNN224_VGG.__name__: model_dev01.SCHEDULER_CHOICES,
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
        model_dev01.FCDD_CNN224_VGG.__name__: model_dev01.LOSS_CHOICES,
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
LIGHTNING_STRATEGY_NONE = "none"
LIGHTNING_STRATEGY_DDP = "ddp"
LIGHTNING_STRATEGY_CHOICES = (
    LIGHTNING_STRATEGY_NONE,
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


# ======================================== wandb ========================================

WANDB_WATCH_NONE = "none"
WANDB_WATCH_GRADIENTS = "gradients"
WANDB_WATCH_ALL = "all"
WANDB_WATCH_PARAMETERS = "parameters"
WANDB_WATCH_CHOICES = (
    WANDB_WATCH_NONE,
    WANDB_WATCH_GRADIENTS,
    WANDB_WATCH_ALL,
    WANDB_WATCH_PARAMETERS,
)
print(f"WANDB_WATCH_CHOICES={WANDB_WATCH_CHOICES}")


WANDB_CHECKPOINT_MODE_NONE = "none"
WANDB_CHECKPOINT_MODE_LAST = "last"
# WANDB_CHECKPOINT_MODE_BEST = "best"
# WANDB_CHECKPOINT_MODE_ALL = "all"
WANDB_CHECKPOINT_MODES = (
    WANDB_CHECKPOINT_MODE_NONE,
    WANDB_CHECKPOINT_MODE_LAST,
    # WANDB_CHECKPOINT_MODE_BEST,
    # WANDB_CHECKPOINT_MODE_ALL, 
)
print(f"WANDB_CHECKPOINT_MODES={WANDB_CHECKPOINT_MODES}")


# ======================================== utills ========================================
       
def _reduce_curve_number_of_points(x, y, npoints) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reduces the number of points in the curve by interpolating linearly.
    Suitable for ROC and PR curves.
    """
    func = interp1d(x, y, kind='linear')
    xmin, xmax = np.min(x), np.max(x)
    xs = np.linspace(xmin, xmax, npoints, endpoint=True)
    return xs, func(xs)
  
  
@torch.no_grad()
def compute_gtmap_pr(
    anomaly_scores,
    original_gtmaps,
    net, 
    limit_npoints: int = 3000,
):
    """
    The scores are upsampled to the images' original size and then the PR is computed.
    The scores are normalized between 0 and 1, and interpreted as anomaly "probability".
    """
    
    # GTMAPS pixel-wise anomaly detection = explanation performance
    print('Computing PR score')
    
    # Reduces the anomaly score to be a score per pixel (explanation)
    anomaly_scores = anomaly_scores.mean(1).unsqueeze(1)
    anomaly_scores = net.receptive_upsample(anomaly_scores, std=net.gauss_std)
        
    # Further upsampling for original dataset size
    anomaly_scores = torch.nn.functional.interpolate(anomaly_scores, (original_gtmaps.shape[-2:]))
    flat_gtmaps, flat_ascores = original_gtmaps.reshape(-1).int().tolist(), anomaly_scores.reshape(-1).tolist()
    
    # ths = thresholds
    precision, recall, ths = precision_recall_curve(
        y_true=flat_gtmaps, 
        probas_pred=flat_ascores,
    )
    
    # a (0, 1) point is added to make the graph look better
    # i discard this because it's not useful and there is no 
    # corresponding threshold 
    precision, recall = precision[:-1], recall[:-1]
    
    # recall must be in descending order 
    # recall = recall[::-1]
    
    # reduce the number of points of the curve
    npoints = ths.shape[0]
    
    if npoints > limit_npoints:
        
        _, precision = _reduce_curve_number_of_points(
            x=ths, 
            y=precision, 
            npoints=limit_npoints,
        )
        ths, recall = _reduce_curve_number_of_points(
            x=ths, 
            y=recall, 
            npoints=limit_npoints,
        )
    
    ap_score = average_precision_score(y_true=flat_gtmaps, y_score=flat_ascores)
    
    print(f'##### GTMAP AP TEST SCORE {ap_score} #####')
    gtmap_pr_res = dict(recall=recall, precision=precision, ths=ths, ap=ap_score)
    return gtmap_pr_res


# ==========================================================================================
# ==========================================================================================
# ======================================== parser ==========================================
# ==========================================================================================
# ==========================================================================================

def parser_add_arguments(parser: ArgumentParser) -> ArgumentParser:
    """
    Defines all the arguments for running an FCDD experiment.
    :param parser: instance of an ArgumentParser.
    :return: the parser with added arguments
    """
    # todo add gradient cliping
    # ===================================== training =====================================
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--learning-rate', type=float)
    parser.add_argument('--weight-decay', type=float)
    # ====================================== model =======================================
    parser.add_argument('--model', type=str, choices=MODEL_CHOICES,)
    parser.add_argument("--loss", type=str, choices=LOSS_CHOICES,)
    parser.add_argument('--optimizer', type=str, choices=OPTIMIZER_CHOICES,)
    parser.add_argument(
        '--scheduler', type=str, choices=SCHEDULER_CHOICES,
        help='The type of learning rate scheduler.'
             '"lambda", reduces the learning rate each epoch by a certain factor.'
    )
    parser.add_argument(
        '--scheduler-paramaters', type=float, nargs='*',
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
    # ===================================== heatmap ======================================
    parser.add_argument(
        '--gauss-std', type=float,
        help='Standard deviation of the Gaussian kernel used for upsampling and blurring.'
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
        '--it', type=int, default=None, 
        help='Number of runs per class with different random seeds. If seeds is specified this is unnecessary.'
    )
    parser.add_argument(
        "--seeds", type=seed_str2int, nargs='*', default=None,
        help="If set, the model will be trained with the given seeds."
            "Otherwise it will be trained with randomly generated seeds."
            "The seeds must be passed in hexadecimal format, e.g. 0x1234."
    ),
    parser.add_argument(
        '--classes', type=int, nargs='+', default=None,
        help='Run only training sessions for some of the classes being nominal. If not give (default) then all classes are trained.'
    )
    parser.add_argument(
        "--cuda-visible-devices", type=int, nargs='*', default=None,
    )
    # ====================================== files =======================================
    parser.add_argument(
        '--logdir', type=Path, default=Path("../../data/results"),
        help='Where log data is to be stored. The start time is put after the dir name. Default: ../../data/results.'
    )
    parser.add_argument('--logdir-suffix', type=str, default='',)
    parser.add_argument('--logdir-prefix', type=str, default='',)
    parser.add_argument(
        '--datadir', type=Path, default=Path("../../data/datasets"),
        help='Where datasets are found or to be downloaded to. Default: ../../data/datasets.',
    )
    # ====================================== wandb =======================================
    parser.add_argument("--wandb-project", type=str,)
    parser.add_argument("--wandb-tags", type=str, nargs='*',)
    parser.add_argument(
        "--wandb-profile", action="store_true",
        help="If set, the run will be profiled and sent to wandb."
    )
    parser.add_argument("--wandb-offline", action="store_true", help="If set, will not sync with the webserver.",)
    parser.add_argument(
        # choices taken from wandb/sdk/wandb_watch.py => watch()
        "--wandb-watch", type=str, default=None, choices=WANDB_WATCH_CHOICES, 
        help="Argument for wandb_logger.watch(..., log=WANDB_WATCH).",
    )
    parser.add_argument(
        "--wandb-watch-log-freq", type=int, default=100,
        help="Log frequency of gradients and parameters. Argument for wandb_logger.watch(..., log_freq=WANDB_WATCH_LOG_FREQ). ",
    )
    parser.add_argument(        
        "--wandb-checkpoint-mode", type=str, choices=WANDB_CHECKPOINT_MODES, default=WANDB_CHECKPOINT_MODE_LAST,
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
        "--wandb-log-score-histogram", type=str, nargs=3, choices=LOG_HISTOGRAM_MODES,
        help="if and how to log  score values; you should give 3 values, respectively for the train/validation/test hooks"
    ),
    parser.add_argument(
        "--wandb-log-loss-histogram", type=str, nargs=3, choices=LOG_HISTOGRAM_MODES,
        help="if and how to log loss values; you should give 3 values, respectively for the train/validation/test hooks"
    ),
    # ================================ pytorch lightning =================================
    parser.add_argument(
        "--lightning-accelerator", type=str, 
        default=LIGHTNING_ACCELERATOR_GPU, 
        choices=LIGHTNING_ACCELERATOR_CHOICES,
        help=f"https://pytorch-lightning.readthedocs.io/en/latest/extensions/accelerator.html"
    )
    parser.add_argument("--lightning-ndevices", type=int, default=1, help="Number of devices (gpus) to use for training.")
    parser.add_argument(
        "--lightning-strategy", type=str, 
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
    
    # ================================== none ==================================
    if args_.lightning_strategy is not None:
        args_.lightning_strategy = None if args_.lightning_strategy == LIGHTNING_STRATEGY_NONE else args_.lightning_strategy

    if args_.wandb_watch is not None:
        args_.wandb_watch = None if args_.wandb_watch == WANDB_WATCH_NONE else args_.wandb_watch

    if args_.wandb_checkpoint_mode is not None:
        args_.wandb_checkpoint_mode = None if args_.wandb_checkpoint_mode == WANDB_CHECKPOINT_MODE_NONE else args_.wandb_checkpoint_mode
      
    # ================================== list 2 tuple (imutable) ==================================
    args_.raw_shape = tuple(args_.raw_shape)
    args_.net_shape = tuple(args_.net_shape)
    
    # ================================== cuda ==================================
    
    if args_.cuda_visible_devices is not None:
        args_.cuda_visible_devices = tuple(args_.cuda_visible_devices)
        
    # ================================== paths ==================================
    args_.datadir = args_.datadir.resolve().absolute()
    
    logdir = args_.logdir.resolve().absolute()
    logdir_name = f"{args_.logdir_prefix}{'_' if args_.logdir_prefix else ''}{logdir.name}_{time_format(args_.script_start_time)}{'_' if args_.logdir_suffix else ''}{args_.logdir_suffix}"
    del vars(args_)['logdir_suffix']
    del vars(args_)['logdir_prefix']
    parent_dir = logdir.parent 
    if args_.wandb_project is not None:
        parent_dir = parent_dir / args_.wandb_project    
    parent_dir = parent_dir / args_.dataset
    args_.logdir = parent_dir / logdir_name
    
    # ================================== seeds ==================================
    seeds = args_.seeds
    it = args_.it
    
    if seeds is None:
        assert it is not None, "seeds or `it` (number of iterations) must be specified"
        print('no seeds specified, using default: auto generated seeds from the system entropy')
        seeds = []
        for _ in range(it):
            seeds.append(create_seed())
            time.sleep(1/3)  # let the system state change
        args_.seeds = tuple(seeds)
    else:
        if args_.it is not None:
            # todo change by alert
            print(f"seeds specified, `it` (={it}) (number of iterations) will be ignored!")
        seeds = tuple(seeds)
        for s in seeds:
            assert type(s) == int, f"seed must be an int, got {type(s)}"
            assert s >= 0, f"seed must be >= 0, got {s}"
        assert len(set(seeds)) == len(seeds), f"seeds must be unique, got {s}"

    del vars(args_)['it']
        
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
    scheduler_paramaters: list,
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
    # heatmap
    gauss_std: float,
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
    wandb_log_roc: Tuple[bool, bool, bool],
    wandb_log_pr: Tuple[bool, bool, bool],
    wandb_log_score_histogram: Tuple[str, str, str],
    wandb_log_loss_histogram: Tuple[str, str, str],
    # pytorch lightning
    lightning_accelerator: str,
    lightning_ndevices: int,
    lightning_strategy: str,
    lightning_precision: str,
    lightning_model_summary_max_depth: int,
    lightning_check_val_every_n_epoch: int,
    lightning_accumulate_grad_batches: int,
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
    
    assert len(wandb_log_roc) == 3, f"wandb_log_roc should have 3 bools, for train/validation/test, but got {wandb_log_roc}" 
    assert all(isinstance(obj, bool) for obj in wandb_log_roc), f"wandb_log_roc should only have bool, got {wandb_log_roc}"
    assert len(wandb_log_pr) == 3, f"wandb_log_pr should have 3 bools, for train/validation/test, but got {wandb_log_pr}" 
    assert all(isinstance(obj, bool) for obj in wandb_log_pr), f"wandb_log_pr should only have bool, got {wandb_log_pr}"
    assert len(wandb_log_score_histogram) == 3, f"wandb_log_score_histogram should have 3 bools, for train/validation/test, but got {len(wandb_log_score_histogram)} things"
    assert all(val in LOG_HISTOGRAM_MODES for val in wandb_log_score_histogram), f"wandb_log_score_histogram values should be in {LOG_HISTOGRAM_MODES}, got {wandb_log_score_histogram}"
    assert len(wandb_log_loss_histogram) == 3, f"wandb_log_loss_histogram should have 3 bools, for train/validation/test, but got {len(wandb_log_loss_histogram)} things"
    assert all(val in LOG_HISTOGRAM_MODES for val in wandb_log_loss_histogram), f"wandb_log_loss_histogram values should be in {LOG_HISTOGRAM_MODES}, got {wandb_log_loss_histogram}"

    logdir.mkdir(parents=True, exist_ok=True)
    
    # seed
    (logdir / "seed.txt").write_text(seed_int2str(seed))
    torch.manual_seed(seed)
    
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
    datamodule.prepare_data()

    # ================================ MODEL ================================
    try:
        model_class = MODEL_CLASSES[model]
       
    except KeyError as err:
        raise NotImplementedError(f'Model {model} is not implemented!') from err

    try:
        model = model_class(
            in_shape=datamodule.net_shape, 
            gauss_std=gauss_std,
            loss_name=loss,
            # optimizer
            optimizer_name=optimizer,
            lr=learning_rate,
            weight_decay=weight_decay,
            # scheduler
            scheduler_name=scheduler,
            scheduler_paramaters=scheduler_paramaters,
        )
    except ModelError as ex:
        msg = ex.args[0]
        if "required positional arguments" in msg:
            raise ModelError(f"Model model_class={model_class.__name__} requires positional argument missing ") from ex
    
    def log_model_architecture(model_: torch.nn.Module):
        model_str = str(model_)
        print(model_str)
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
    ]
    
    log_roc_train, log_roc_validation, log_roc_test = wandb_log_roc
    # ========================================================================= ROC
    
    if log_roc_train:
        callbacks.append(
            LogRocCallback(
                stage=RunningStage.TRAINING,
                scores_key="score_maps",
                gt_key="gtmaps",
                log_curve=False,
                limit_points=3000,  # todo make it script param
                python_generator=create_python_random_generator(seed),
            )
        )
        
    if log_roc_validation:
        callbacks.append(
            LogRocCallback(
                stage=RunningStage.VALIDATING,
                scores_key="score_maps",
                gt_key="gtmaps",
                log_curve=False,
                limit_points=3000,  # todo make it script param
                python_generator=create_python_random_generator(seed),
            )
        )
        
    if log_roc_test:
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
    # ========================================================================= PR
    log_pr_train, log_pr_validation, log_pr_test = wandb_log_pr
    
    if log_pr_train:
        callbacks.append(
            LogAveragePrecisionCallback(
                stage=RunningStage.TRAINING,
                scores_key="score_maps",
                gt_key="gtmaps",
                log_curve=False,
                limit_points=3000,  # todo make it script param
                python_generator=create_python_random_generator(seed),
            )
        )
    if log_pr_validation:
        callbacks.append(
            LogAveragePrecisionCallback(
                stage=RunningStage.VALIDATING,
                scores_key="score_maps",
                gt_key="gtmaps",
                log_curve=False,
                limit_points=3000,  # todo make it script param
                python_generator=create_python_random_generator(seed),
            )
        )
    if log_pr_test:
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
    # ========================================================================= heatmaps
    
    
    # ========================================================================= histograms
    
    log_score_histogram_train, log_score_histogram_validation, log_score_histogram_test = wandb_log_score_histogram
    
    if log_score_histogram_train != LOG_HISTOGRAM_MODE_NONE:
        callbacks.extend([
            LogHistogramCallback(stage=RunningStage.TRAINING, key="score_maps", mode=log_score_histogram_train,), 
            LogHistogramsSuperposedPerClassCallback(
                stage=RunningStage.TRAINING, 
                values_key="score_maps", 
                gt_key="gtmaps", 
                mode=log_score_histogram_train,
                python_generator=create_python_random_generator(seed),
                limit_points=3000,  # 3k
            ),
        ])

    if log_score_histogram_validation != LOG_HISTOGRAM_MODE_NONE:
        
        callbacks.extend([
            LogHistogramCallback(stage=RunningStage.VALIDATING, key="score_maps", mode=log_score_histogram_validation,), 
            LogHistogramsSuperposedPerClassCallback(
                stage=RunningStage.VALIDATING, 
                values_key="score_maps", 
                gt_key="gtmaps", 
                mode=log_score_histogram_validation,
                python_generator=create_python_random_generator(seed),
                limit_points=3000,  # 3k
            ),
        ])
    
    if log_score_histogram_test != LOG_HISTOGRAM_MODE_NONE:
        callbacks.extend([
            LogHistogramCallback(stage=RunningStage.TESTING, key="score_maps", mode=log_score_histogram_test,), 
            LogHistogramsSuperposedPerClassCallback(
                stage=RunningStage.TESTING, 
                values_key="score_maps", 
                gt_key="gtmaps", 
                mode=log_score_histogram_test,
                python_generator=create_python_random_generator(seed),
                limit_points=30000,  # 30k
            ),
        ])
    
    log_loss_histogram_train, log_loss_histogram_validation, log_loss_histogram_test = wandb_log_loss_histogram
    
    if log_loss_histogram_train != LOG_HISTOGRAM_MODE_NONE:
        callbacks.extend([
            LogHistogramCallback(stage=RunningStage.TRAINING, key="loss_maps", mode=log_loss_histogram_train,), 
            LogHistogramsSuperposedPerClassCallback(
                stage=RunningStage.TRAINING, 
                values_key="loss_maps", 
                gt_key="gtmaps", 
                mode=log_loss_histogram_train,
                python_generator=create_python_random_generator(seed),
                limit_points=3000,  # 3k
            ),
        ])
    
    if log_loss_histogram_validation != LOG_HISTOGRAM_MODE_NONE:
        callbacks.extend([
            LogHistogramCallback(stage=RunningStage.VALIDATING, key="loss_maps", mode=log_loss_histogram_validation,), 
            LogHistogramsSuperposedPerClassCallback(
                stage=RunningStage.VALIDATING, 
                values_key="loss_maps", 
                gt_key="gtmaps", 
                mode=log_loss_histogram_validation,
                python_generator=create_python_random_generator(seed),
                limit_points=3000,  # 3k
            ),
        ])
    
    if log_loss_histogram_test != LOG_HISTOGRAM_MODE_NONE:
        callbacks.extend([
            LogHistogramCallback(stage=RunningStage.TESTING, key="loss_maps", mode=log_loss_histogram_test,), 
            LogHistogramsSuperposedPerClassCallback(
                stage=RunningStage.TESTING, 
                values_key="loss_maps", 
                gt_key="gtmaps", 
                mode=log_loss_histogram_test,
                python_generator=create_python_random_generator(seed),
                limit_points=30000,  # 30k
            ),
        ])

    # if preview_nimages > 0:
    #     datamodule.setup("fit")
    #     callbacks.append(
    #         DataloaderPreviewCallback(
    #             dataloader=datamodule.train_dataloader(embed_preprocessing=True), 
    #             n_samples=preview_nimages, logkey_prefix="train-preview"
    #         ),
    #     )
    
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

    # ================================ FIT ================================
    trainer = pl.Trainer(
        accelerator=lightning_accelerator,
        gpus=lightning_ndevices, 
        strategy=lightning_strategy,
        precision=lightning_precision,
        logger=wandb_logger,  
        log_every_n_steps=1,  
        max_epochs=epochs,    
        deterministic=True,
        callbacks=callbacks, 
        # i should learn how to properly deal with
        # the sanity check but for now it's causing too much trouble
        num_sanity_val_steps=0,
        # todo add accumulate_grad_batches
        # todo add auto_scale_batch_size
        # chek deterministic in detail
        # todo make specific callbacks on LightningModule.configure_callbacks(),
        check_val_every_n_epoch=lightning_check_val_every_n_epoch,
        accumulate_grad_batches=lightning_accumulate_grad_batches,
    )
    
    with profiler:
        datamodule.setup("fit")
        trainer.fit(model=model, datamodule=datamodule)
    
    if wandb_watch is not None:
        wandb_logger.experiment.unwatch(model)

    if wandb_profile:
        profile_art = wandb.Artifact(f"trace-{wandb.run.id}", type="profile")
        profile_art.add_file(str(next(profile_dir.glob("*.pt.trace.json"))), "trace.pt.trace.json")
        wandb_logger.experiment.log_artifact(profile_art)

    if not test:
        return 
    
    # ================================ TEST ================================
    trainer.test(model=model, datamodule=datamodule)    
        
# ==========================================================================================
# ==========================================================================================
# ========================================== run ===========================================
# ==========================================================================================
# ==========================================================================================

    
def run(**kwargs) -> dict:
    """see the arguments in run_one()"""
    
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
    
    if wandb_checkpoint_mode is None:
        log_model = False 
    else:
        log_model = wandb_checkpoint_mode != WANDB_CHECKPOINT_MODE_NONE
        
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
                    normal_class_label=dataset_class_labels(dataset)[c],
                    script_start_time=script_start_time,
                    it=it,
                    cuda_visible_devices=cuda_visible_devices,
                ),
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
            
            try:
                run_one(wandb_logger=wandb_logger, **run_one_kwargs,)
            
            except TypeError as ex:
                msg = ex.args[0]
                if "run_one() got an unexpected keyword argument" in msg:
                    raise ScriptError(f"run_one() got an unexpected keyword argument: {msg}, did you forget to kwargs.pop() something?") from ex
                raise ex
            except Exception as ex:
                wandb_logger.finalize("failed")
                wandb.finish(1)
                raise ex
            else:
                wandb_logger.finalize("success")
                wandb.finish(0)
                