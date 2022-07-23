#!/usr/bin/env python
# coding: utf-8
import functools
import itertools
from pathlib import Path
from re import A
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import warnings

import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.profiler import (AdvancedProfiler, PyTorchProfiler,
                                        SimpleProfiler)

import model_dev01_bis
import mvtec_dataset_dev01_bis as mvtec_dataset_dev01_bis
import wandb
from callbacks_dev01_bis import (DataloaderPreviewCallback, LearningRateLoggerCallback)
from common_dev01_bis import (ArgumentParserOrArgumentGroup, hashify_config,
                              none_or_str, seed_int2str)
from pytorch_lightning.callbacks import StochasticWeightAveraging

# ======================================== lightning ========================================

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

LIGHTNING_SWA_ANNEALING_STRATEGY_LINEAR = "linear"
LIGHTNING_SWA_ANNEALING_STRATEGY_COS = "cos"
LIGHTNING_SWA_ANNEALING_STRATEGY_CHOICES = (
    LIGHTNING_SWA_ANNEALING_STRATEGY_LINEAR, LIGHTNING_SWA_ANNEALING_STRATEGY_COS,
)

# ======================================== wandb ========================================

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

# ======================================== dataset ========================================

DATASET_CHOICES = (mvtec_dataset_dev01_bis.DATASET_NAME,)
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
        mvtec_dataset_dev01_bis.DATASET_NAME: mvtec_dataset_dev01_bis.CLASSES_LABELS,
    }[dataset_name]
    

@unknown_dataset
def dataset_class_fullqualified(dataset_name: str) -> List[str]:
    return {
        mvtec_dataset_dev01_bis.DATASET_NAME: mvtec_dataset_dev01_bis.CLASSES_FULLQUALIFIED,
    }[dataset_name]


@unknown_dataset
def dataset_nclasses(dataset_name: str) -> int:
    return {
        mvtec_dataset_dev01_bis.DATASET_NAME: mvtec_dataset_dev01_bis.NCLASSES,
    }[dataset_name]


@unknown_dataset
def dataset_class_index(dataset_name: str, class_name: str) -> int:
    return dataset_class_labels(dataset_name).index(class_name)

# ======================================== preprocessing ========================================

@unknown_dataset
def dataset_preprocessing_choices(dataset_name: str) -> List[str]:
    return {
        mvtec_dataset_dev01_bis.DATASET_NAME: mvtec_dataset_dev01_bis.PREPROCESSING_CHOICES,
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
        mvtec_dataset_dev01_bis.DATASET_NAME: mvtec_dataset_dev01_bis.SUPERVISE_MODES,
    }[dataset_name]


ALL_SUPERVISE_MODE_CHOICES = tuple(set.union(*[
    set(dataset_supervise_mode_choices(dataset_name)) 
    for dataset_name in DATASET_CHOICES
]))
print(f"SUPERVISE_MODE_CHOICES={ALL_SUPERVISE_MODE_CHOICES}")

# ======================================== models ========================================

MODEL_CLASSES = {
    model_dev01_bis.MODEL_FCDD_CNN224_VGG_F: model_dev01_bis.FCDD,
    model_dev01_bis.MODEL_U2NET_HEIGHT4_LITE: model_dev01_bis.HyperSphereU2Net,
    model_dev01_bis.MODEL_U2NET_HEIGHT6_LITE: model_dev01_bis.HyperSphereU2Net,
    model_dev01_bis.MODEL_U2NET_HEIGHT6_FULL: model_dev01_bis.HyperSphereU2Net,
    model_dev01_bis.MODEL_U2NET_REPVGG_HEIGHT4_LITE: model_dev01_bis.HyperSphereU2Net,
    model_dev01_bis.MODEL_U2NET_REPVGG_HEIGHT5_FULL: model_dev01_bis.HyperSphereU2Net,
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
        model_dev01_bis.MODEL_FCDD_CNN224_VGG_F: model_dev01_bis.OPTIMIZER_CHOICES,
        model_dev01_bis.MODEL_U2NET_HEIGHT4_LITE: model_dev01_bis.OPTIMIZER_CHOICES,
        model_dev01_bis.MODEL_U2NET_HEIGHT6_LITE: model_dev01_bis.OPTIMIZER_CHOICES,
        model_dev01_bis.MODEL_U2NET_HEIGHT6_FULL: model_dev01_bis.OPTIMIZER_CHOICES,
        model_dev01_bis.MODEL_U2NET_REPVGG_HEIGHT4_LITE: model_dev01_bis.OPTIMIZER_CHOICES,
        model_dev01_bis.MODEL_U2NET_REPVGG_HEIGHT5_FULL: model_dev01_bis.OPTIMIZER_CHOICES,
    }[model_name]   


OPTIMIZER_CHOICES = tuple(set.union(*[
    set(model_optimizer_choices(model_name))
    for model_name in MODEL_CHOICES
]))
print(f"OPTIMIZER_CHOICES={OPTIMIZER_CHOICES}")  

# ======================================== dropout ========================================

@unknown_model
def model_dropout_mode_choices(model_name: str) -> List[str]:
    return {
        # fcdd
        model_dev01_bis.MODEL_FCDD_CNN224_VGG_F: model_dev01_bis.DROPOUT_MODE_FCDD_CHOICES,
        # u2net
        model_dev01_bis.MODEL_U2NET_HEIGHT4_LITE: model_dev01_bis.DROPOUT_MODE_U2NET_CHOICES,
        model_dev01_bis.MODEL_U2NET_HEIGHT6_LITE: model_dev01_bis.DROPOUT_MODE_U2NET_CHOICES,
        model_dev01_bis.MODEL_U2NET_HEIGHT6_FULL: model_dev01_bis.DROPOUT_MODE_U2NET_CHOICES,
        model_dev01_bis.MODEL_U2NET_REPVGG_HEIGHT4_LITE: model_dev01_bis.DROPOUT_MODE_U2NET_CHOICES,
        model_dev01_bis.MODEL_U2NET_REPVGG_HEIGHT5_FULL: model_dev01_bis.DROPOUT_MODE_U2NET_CHOICES,
    }[model_name]


DROPOUT_MODE_CHOICES = tuple(set.union(*[
    set(model_dropout_mode_choices(model_name))
    for model_name in MODEL_CHOICES
]))
print(f"DROPOUT_MODE_CHOICES={DROPOUT_MODE_CHOICES}")


# ======================================== schedulers ========================================

@unknown_model
def model_scheduler_choices(model_name: str) -> List[str]:
    return {
        model_dev01_bis.MODEL_FCDD_CNN224_VGG_F: model_dev01_bis.SCHEDULER_CHOICES,
        model_dev01_bis.MODEL_U2NET_HEIGHT4_LITE: model_dev01_bis.SCHEDULER_CHOICES,
        model_dev01_bis.MODEL_U2NET_HEIGHT6_LITE: model_dev01_bis.SCHEDULER_CHOICES,
        model_dev01_bis.MODEL_U2NET_HEIGHT6_FULL: model_dev01_bis.SCHEDULER_CHOICES,
        model_dev01_bis.MODEL_U2NET_REPVGG_HEIGHT4_LITE: model_dev01_bis.SCHEDULER_CHOICES,
        model_dev01_bis.MODEL_U2NET_REPVGG_HEIGHT5_FULL: model_dev01_bis.SCHEDULER_CHOICES,
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
        model_dev01_bis.MODEL_FCDD_CNN224_VGG_F: model_dev01_bis.LOSS_FCDD_CHOICES,
        model_dev01_bis.MODEL_U2NET_HEIGHT4_LITE: model_dev01_bis.LOSS_U2NET_CHOICES,
        model_dev01_bis.MODEL_U2NET_HEIGHT6_LITE: model_dev01_bis.LOSS_U2NET_CHOICES,
        model_dev01_bis.MODEL_U2NET_HEIGHT6_FULL: model_dev01_bis.LOSS_U2NET_CHOICES,
        model_dev01_bis.MODEL_U2NET_REPVGG_HEIGHT4_LITE: model_dev01_bis.LOSS_U2NET_CHOICES,
        model_dev01_bis.MODEL_U2NET_REPVGG_HEIGHT5_FULL: model_dev01_bis.LOSS_U2NET_CHOICES,
    }[model_name]


LOSS_CHOICES = tuple(set.union(*[
    set(model_loss_choices(model_name))
    for model_name in MODEL_CHOICES
]))
print(f"LOSS_CHOICES={LOSS_CHOICES}")

# ======================================== exceptions ========================================

class ScriptError(Exception):
    pass


class ModelError(Exception):
    pass


# ==========================================================================================
# ==========================================================================================
# ======================================== parser ==========================================
# ==========================================================================================
# ==========================================================================================

def cli_add_arguments_run(parser: ArgumentParserOrArgumentGroup):
    parser.add_argument("--wandb_issweep", action="store_true", help="use this when you are launching a sweep")
    parser.add_argument("--wandb_entity", type=str,)
    parser.add_argument("--wandb_project", type=str,)
    parser.add_argument(
        '--classes', type=int, nargs='+', 
        help='Run only training sessions for some of the classes being nominal. If not give (default) then all classes are trained.'
    )
    
    
def args_run_validate_wandb_options(args):
    
    if args.wandb_issweep:
        if args.wandb_entity is not None:
            warnings.warn("--wandb_entity is ignored when --wandb_issweep is set")
        if args.wandb_project is not None:
            warnings.warn("--wandb_project is ignored when --wandb_issweep is set")
    else:
        assert args.wandb_entity is not None, f"--wandb_entity is required when --wandb_issweep is not set"
        assert args.wandb_project is not None, f"--wandb_project is required when --wandb_issweep is not set"


def cli_add_arguments_run_one(parser: ArgumentParserOrArgumentGroup):
    # ===================================== training =====================================
    parser.add_argument('--epochs', type=int,)
    parser.add_argument('--learning_rate', type=float,)
    parser.add_argument('--weight_decay', type=float,)
    parser.add_argument(
        '--no_test', dest="test", action="store_false",
        help='If set then the model will not be tested at the end of the training. It will by default.'
    )
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
        '--scheduler_parameters', type=float, nargs='*',
        help='Sequence of learning rate scheduler parameters. '
             '"lambda": one parameter is allowed, the factor the learning rate is reduced per epoch. '
    )
    parser.add_argument("--dropout_mode", type=str, choices=DROPOUT_MODE_CHOICES,)
    parser.add_argument("--dropout_parameters", type=float, nargs='*',)
    # ====================================== dataset =====================================
    parser.add_argument('--dataset', type=str, choices=DATASET_CHOICES,)
    parser.add_argument("--raw_shape", type=int, nargs=2,)
    parser.add_argument("--net_shape", type=int, nargs=2,)
    parser.add_argument('--batch_size', type=int,)
    parser.add_argument(
        '--nworkers', type=int, 
        help='Number of workers for data loading (DataLoader parameter).'
    )
    parser.add_argument('--pin_memory', action='store_true', )
    parser.add_argument(
        '--preprocessing', type=none_or_str, choices=ALL_PREPROCESSING_CHOICES,
        help='Preprocessing pipeline (augmentations and such). Defined inside each dataset module.'
    )
    parser.add_argument(
        "--preprocess_moment", type=str, choices=mvtec_dataset_dev01_bis.DATAMODULE_PREPROCESS_MOMENT_CHOICES,
        help="Should the preprocessing be applied before or after the data being transferred to the GPU?",
    )
    parser.add_argument(
        '--supervise_mode', type=str, choices=ALL_SUPERVISE_MODE_CHOICES,
        help='This determines the kind of artificial anomalies. '
    )
    parser.add_argument(
        '--real_anomaly_limit', type=int,
        help='Determines the number of real anomalous images used in the case of real anomaly supervision. '
             'Has no impact on synthetic anomalies.'
    )
    parser.add_argument(
        '--datadir', type=Path,
        help='Where datasets are found or to be downloaded to. Default: ../../data/datasets.',
    )
    # ================================ pytorch lightning =================================
    parser.add_argument(
        "--lightning_accelerator", type=str, choices=LIGHTNING_ACCELERATOR_CHOICES,
        help=f"https://pytorch-lightning.readthedocs.io/en/latest/extensions/accelerator.html"
    )
    parser.add_argument(
        "--lightning_ndevices", type=int, 
        help="Number of devices (gpus) to use for training."
    )
    parser.add_argument(
        "--lightning_strategy", type=none_or_str, choices=LIGHTNING_STRATEGY_CHOICES,
        help="See https://pytorch-lightning.readthedocs.io/en/latest/extensions/strategy.html"
    )
    parser.add_argument(
        "--lightning_precision", type=int, choices=LIGHTNING_PRECISION_CHOICES,
        help="https://pytorch-lightning.readthedocs.io/en/latest/guides/speed.html#mixed-precision-16-bit-training"
    )
    parser.add_argument(
        "--lightning_model_summary_max_depth", type=int,
        help="https://pytorch-lightning.readthedocs.io/en/latest/api/pytorch_lightning.callbacks.ModelSummary.html#pytorch_lightning.callbacks.ModelSummary",
    )
    parser.add_argument(
        "--lightning_check_val_every_n_epoch", type=int,
        help="... find link to lightning doc ..."
    )
    parser.add_argument(
        "--lightning_accumulate_grad_batches", type=int,
        help="Accumulate gradients for THIS batches. https://pytorch-lightning.readthedocs.io/en/latest/advanced/training_tricks.html#accumulate-gradients"
    )
    parser.add_argument(
        "--lightning_profiler", type=none_or_str, choices=LIGHTNING_PROFILER_CHOICES,
        help="simple and advanced: https://pytorch-lightning.readthedocs.io/en/latest/tuning/profiler_basic.html\n"
             "pytorch: https://pytorch-lightning.readthedocs.io/en/latest/tuning/profiler_intermediate.html\n"
             "in any case it is saved in a f"
    )
    group_gradient_clipping = parser.add_argument_group("gradient-clipping")
    group_gradient_clipping.add_argument(
        "--lightning_gradient_clip_val", type=float,
        help=f"https://pytorch-lightning.readthedocs.io/en/latest/advanced/training_tricks.html#gradient-clipping",
    )
    group_gradient_clipping.add_argument(
        "--lightning_gradient_clip_algorithm", type=str, choices=LIGHTNING_GRADIENT_CLIP_ALGORITHM_CHOICES,
        help=f"https://pytorch-lightning.readthedocs.io/en/latest/advanced/training_tricks.html#gradient-clipping",
    )
    parser.add_argument(
        "--lightning_deterministic", type=bool, 
        help="https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html#reproducibility",
    )
    
    # ================================ pytorch lightning SWA =================================
    
    group_swa = parser.add_argument_group("swa", description="Stochastic Weight Averaging")
    group_swa.add_argument(
        "--lightning_swa_enabled", action='store_true',
        help="Activate Stochastic Weight Averaging (SWA). See https://pytorch-lightning.readthedocs.io/en/latest/api/pytorch_lightning.callbacks.StochasticWeightAveraging.html#pytorch_lightning.callbacks.StochasticWeightAveraging"
    )
    group_swa.add_argument("--lightning_swa_learning_rate", type=float,)
    group_swa_epoch_start = group_swa.add_mutually_exclusive_group()
    group_swa_epoch_start.add_argument("--lightning_swa_epoch_start_absolute", dest="lightning_swa_epoch_start", type=int,)
    group_swa_epoch_start.add_argument("--lightning_swa_epoch_start_relative", dest="lightning_swa_epoch_start", type=float,)
    group_swa.add_argument("--lightning_swa_annealing_epochs", type=int,)
    group_swa.add_argument("--lightning_swa_annealing_strategy", type=str, choices=LIGHTNING_SWA_ANNEALING_STRATEGY_CHOICES,)


# ====================================== minimal validations for cli arguments =====================================

def args_validate_dataset_specific_choices(args):
    
    assert args.dataset in DATASET_CHOICES, f"Invalid dataset: {args.dataset}, chose from {DATASET_CHOICES}"
    
    # dataset-specific validation
    assert args.preprocessing in dataset_preprocessing_choices(args.dataset), f"args.dataset={args.dataset}: {args.preprocessing} not in {dataset_preprocessing_choices(args.dataset)}"
    assert args.supervise_mode in dataset_supervise_mode_choices(args.dataset), f"args.dataset={args.dataset}: {args.supervise_mode} not in {dataset_supervise_mode_choices(args.dataset)}"


def args_validate_model_specific_choices(args):
    
    assert args.model in MODEL_CHOICES, f"Invalid model: {args.model}, chose from {MODEL_CHOICES}"
    
    # model-specific validation   
    assert args.loss in model_loss_choices(args.model), f"args.model={args.model}: {args.loss} not in {model_loss_choices(args.model)}"
    assert args.optimizer in model_optimizer_choices(args.model), f"args.model={args.model}: {args.optimizer} not in {model_optimizer_choices(args.model)}"
    assert args.scheduler in model_scheduler_choices(args.model), f"args.model={args.model}: {args.scheduler} not in {model_scheduler_choices(args.model)}"
    assert args.dropout_mode in model_dropout_mode_choices(args.model), f"args.model={args.model}: {args.dropout_mode} not in {model_dropout_mode_choices(args.model)}"

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
    test: bool,
    # model
    model: str, 
    loss: str,
    optimizer: str, 
    scheduler: str,
    scheduler_parameters: list,
    dropout_mode: str,
    dropout_parameters: list,
    # dataset
    dataset: str,
    raw_shape: Tuple[int, int],
    net_shape: Tuple[int, int],
    batch_size: int,
    nworkers: int, 
    pin_memory: bool,
    preprocessing: str,
    preprocess_moment: str,
    supervise_mode: str,
    real_anomaly_limit: int,
    datadir: Path,
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
    # stochastic weight averaging
    lightning_swa_enabled: bool,
    lightning_swa_learning_rate: Optional[float],
    lightning_swa_epoch_start: Optional[Union[int, float]],
    lightning_swa_annealing_epochs: Optional[int],
    lightning_swa_annealing_strategy: Optional[str],
    # from run()
    rundir: Path,
    normal_class: int,
    seed: int,
    #
    wandb_logger: WandbLogger,
    callbacks: list,
):
    # lightning-stuff validation
    assert lightning_accelerator in LIGHTNING_ACCELERATOR_CHOICES, f"Invalid lightning_accelerator: {lightning_accelerator}, chose from {LIGHTNING_ACCELERATOR_CHOICES}"
    if lightning_strategy is not None:
        assert lightning_strategy in LIGHTNING_STRATEGY_CHOICES, f"Invalid lightning_strategy: {lightning_strategy}, chose from {LIGHTNING_STRATEGY_CHOICES}"
    assert lightning_precision in LIGHTNING_PRECISION_CHOICES, f"Invalid lightning_precision: {lightning_precision}, chose from {LIGHTNING_PRECISION_CHOICES}"
    assert lightning_profiler in LIGHTNING_PROFILER_CHOICES, f"Invalid lightning_profiler: {lightning_profiler}, chose from {LIGHTNING_PROFILER_CHOICES}"

    rundir.mkdir(parents=True, exist_ok=True)
    
    # seed
    (rundir / "seed.txt").write_text(seed_int2str(seed))
    torch.manual_seed(seed)
    
    batch_size_effective = batch_size * lightning_accumulate_grad_batches if lightning_accumulate_grad_batches > 0 else batch_size
    
    summary_update_dict_inputs = dict(
        batch_size_effective=batch_size_effective,
    )
    wandb.run.summary.update(summary_update_dict_inputs)
    
    # ================================ DATA ================================
    
    print(f"datadir: from run_one() args: {datadir}")
    datadir = datadir.resolve().absolute()
    print(f"datadir: resolved: {datadir}")
    
    # datamodule hard-coded for now, but later other datasets can be added
    datamodule = mvtec_dataset_dev01_bis.MVTecAnomalyDetectionDataModule(
        root=datadir,
        normal_class=normal_class,
        preprocessing=preprocessing,
        preprocess_moment=preprocess_moment,
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
        mvtec_class_type=mvtec_dataset_dev01_bis.CLASSES_TYPES[normal_class],
        normal_class_fullqualified=dataset_class_fullqualified(dataset)[normal_class],
    )
    wandb.run.summary.update(summary_update_dict_dataset)
    datamodule.prepare_data()
    datamodule.setup()

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
            model_name=model,
            # scheduler
            scheduler_name=scheduler,
            scheduler_parameters=scheduler_parameters,
            # dropout
            dropout_mode=dropout_mode,
            dropout_parameters=dropout_parameters,
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
      
    if model_class == model_dev01_bis.HyperSphereU2Net:  
        test_input_output_dimensions(datamodule.net_shape)
    
    def log_model_architecture(model_: torch.nn.Module):
        model_str = str(model_)
        model_str_fpath = rundir / "model.txt"
        model_str_fpath.write_text(model_str)
         # now = dont keep syncing if it changes
        print(f"saving model architecture to {str(model_str_fpath)}")
        wandb.save(str(model_str_fpath), policy="now", base_path=str(model_str_fpath.parent)) 
    
    log_model_architecture(model)
        
    # ================================ CALLBACKS ================================

    callbacks = [
        # pl.callbacks.ModelSummary(max_depth=lightning_model_summary_max_depth),
        pl.callbacks.RichModelSummary(max_depth=lightning_model_summary_max_depth),
        LearningRateLoggerCallback(),
        DataloaderPreviewCallback(
            datamodule.train_dataloader(batch_size_override=10, nworkers_override=0, embed_preprocessing=True),
            n_samples=20,
            stage="train",
        ),
        # DataloaderPreviewCallback(
        #     datamodule.val_dataloader(batch_size_override=10, embed_preprocessing=True),
        #     n_samples=20,
        #     stage="validate",
        # ),
    ] + callbacks
    
    # ================================ SWA ================================
    if lightning_swa_enabled:
        assert lightning_swa_learning_rate is not None, f"lightning_swa_learning_rate must be set if lightning_swa_enabled is True"
        assert lightning_swa_epoch_start is not None, f"lightning_swa_epoch_start must be set if lightning_swa_enabled is True"
        assert lightning_swa_annealing_epochs is not None, f"lightning_swa_annealing_epochs must be set if lightning_swa_enabled is True"
        assert lightning_swa_annealing_strategy is not None, f"lightning_swa_annealing_strategy must be set if lightning_swa_enabled is True"
        
        assert isinstance(lightning_swa_learning_rate, float), f"lightning_swa_learning_rate must be a float, found {type(lightning_swa_learning_rate)}"
        assert lightning_swa_learning_rate > 0, f"lightning_swa_learning_rate must be > 0, found {lightning_swa_learning_rate}"
        assert 0 < lightning_swa_epoch_start < epochs, f"lightning_swa_epoch_start must be > 0 and < epochs, found {lightning_swa_epoch_start} and {epochs}"
        
        callbacks.append(StochasticWeightAveraging(
            swa_lrs=lightning_swa_learning_rate, 
            swa_epoch_start=lightning_swa_epoch_start, 
            annealing_epochs=lightning_swa_annealing_epochs, 
            annealing_strategy=lightning_swa_annealing_strategy, 
            avg_fn=None, 
            device=None, 
        ))
    
    # ================================ PROFILING ================================
        
    def get_lightning_profiler(profiler_choice: Optional[str]):
        
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

    lightning_profiler = get_lightning_profiler(lightning_profiler)
    
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
        profiler=lightning_profiler,
        gradient_clip_val=lightning_gradient_clip_val,
        gradient_clip_algorithm=lightning_gradient_clip_algorithm,
        deterministic=lightning_deterministic,
    )
    
    trainer.fit(model=model, datamodule=datamodule)
    
    if lightning_profiler is not None:
        profile_fpath = Path(lightning_profiler.dirpath) / lightning_profiler.filename
        print(f"Saving profiler {lightning_profiler} at {profile_fpath}")
        wandb.save(str(profile_fpath), policy="now", base_path=str(profile_fpath.parent)) 

    # ================================ TEST ================================
    if test:
        trainer.test(model=model, datamodule=datamodule)    
        
        
        
# ==========================================================================================
# ==========================================================================================
# ========================================== run ===========================================
# ==========================================================================================
# ==========================================================================================

    
def run(
    wandb_issweep: bool,
    wandb_entity: str,
    wandb_project: str,
    wandb_offline: bool,
    wandb_tags: List[str],
    start_time: int,
    base_rundir: Path,
    seeds: List[int],
    classes: List[str],
    runone_common_kwargs: dict = {},
    confighashes_keys: Dict[str, Tuple[str, ...]] = {},
    wandb_init_config_extra: dict = {},
    callbacks: List[pl.Callback] = [],
) -> dict:
    
    # if none then do all the classes
    classes = classes or tuple(range(dataset_nclasses(runone_common_kwargs['dataset'])))
    its = tuple(range(len(seeds)))
    
    for c, (it, seed) in itertools.product(classes, zip(its, seeds)):
        print(f"class {c:02d}")
        print(f"it {it:02d} seed {seed_int2str(seed)}")    

        rundir = base_rundir / f'normal_{c:02d}' / f'it_{it:02}_seed{seed_int2str(seed)}'
        rundir = rundir.resolve().absolute()
        print(f"rundir: {rundir}")
        
        # it's super important that the dir must already exist for wandb logging           
        rundir.mkdir(parents=True, exist_ok=True)
        
        # ==========================================================================================
        # ==========================================================================================
        # disclaimer: this part is a bit confusing and the order of things is very important!!!
        # ==========================================================================================
        # ==========================================================================================
        
        run_one_kwargs = {
            **runone_common_kwargs, 
            **dict(
                rundir=rundir,  
                seed=seed,
                normal_class=c,
            )
        }
        print(f"run_one_kwargs: {run_one_kwargs}")
        
        # the ones added here don't go to the run_one()            
        wandb_config = {
            **run_one_kwargs,
            **dict(
                seed_str=seed_int2str(seed),
                script_start_time=start_time,
                it=it,
                run_kwargs_classes=classes,
                callbacks=[str(cb) for cb in callbacks],
            ),
            **wandb_init_config_extra,
        }
        print(f"wandb_config: {wandb_config}")
        
        # add hashes to the make it easier to group things in wandb
        confighashes = {
            name: dict(
                desc=f"hash of: ({', '.join(configkeys)})",
                value=hashify_config(config_dict=wandb_config, keys=configkeys, ),
            )
            for name, configkeys in confighashes_keys.items()  
        }
        print(f"confighashes: {confighashes}")
        
        wandb_init_kwargs_noconfig = dict(      
            tags=wandb_tags,
            save_code=True,
            reinit=True,
        )
        if not wandb_issweep:
            
            wandb_init_kwargs_noconfig = dict(
                entity=wandb_entity,
                project=wandb_project, 
                **wandb_init_kwargs_noconfig,
            )
        
        print(f"wandb_init_kwargs_noconfig: {wandb_init_kwargs_noconfig}")
        
        wandblogger_init_kwargs = dict(
            **wandb_init_kwargs_noconfig,
            config=dict(**wandb_config, **confighashes),
        )
                
        wandb_logger = WandbLogger(
            save_dir=str(rundir),
            offline=wandb_offline,
            log_model=not wandb_offline,  # save only the last weights
            **wandblogger_init_kwargs,
        )
        
        wandb_logger_save_dir = Path(wandb_logger.save_dir).resolve().absolute()
        print(f"wandb_logger_save_dir: {wandb_logger_save_dir}")
        
        wandb_init_kwargs = dict(**wandblogger_init_kwargs)
        
        if not wandb_issweep:
            # make sure both have the same run_id
            wandb_init_kwargs["id"] = wandb_logger.experiment._run_id
            
        # image logging is not working properly with the logger
        # so i also use the default wandb interface for that
        # dir: equivalent of savedir in the logger
        run = wandb.init(dir=str(rundir), **wandb_init_kwargs,)
        
        try:            
            run_one(wandb_logger=wandb_logger, callbacks=callbacks, **run_one_kwargs,)
        
        except TypeError as ex:
            run.tags = run.tags + ("crashed",)
            
            wandb_logger.finalize("crashed")
            wandb.finish(1)
            
            msg = ex.args[0]
            
            if "run_one() got an unexpected keyword argument" in msg:
                raise ScriptError(f"run_one() got an unexpected keyword argument: {msg}, did you forget to kwargs.pop() something?") from ex
            
            raise ex
        
        except Exception as ex:
            run.tags = run.tags + ("crashed",)
            
            wandb_logger.finalize("crashed")
            wandb.finish(1)
            
            raise ex
        
        else:
            wandb_logger.finalize("success")
            wandb.finish(0)
            
        finally:
            print(f"wandb_logger_save_dir: {wandb_logger_save_dir}")
            