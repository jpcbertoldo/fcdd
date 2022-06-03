"""

next
next
next
next
next
next
next
next
next
next
next
next
next
next
next
next
next
next
next

for roc callback
    add option to reduce points in the roc curve not samples
    check if necessary!!!! seems like it already does that
    
add a roc callback without sampling in test

clean last_epoch_outputs to mixin
move callbacks to a new python file
make a pr callback like roc
make a callback to plot the segmentations ==> get rid of preview callback, just make the preview callback to plot the segmentations on the first batch...
scores distribution call back

config all these from script cli ==> control frequency !

clean PYTORCH_LIGHTNING_STAGE conts to use the enum from the module

later: separate the different modes in the online replacer ==> online replacer should be a callback!!! (can lightning data module have callbacks?)
later: t-sne of embeddings callback

"""
#!/usr/bin/env python
# coding: utf-8

import contextlib
import functools
import json
import os
import os.path as pt
from re import A
import sys
import time
from argparse import ArgumentError, ArgumentParser
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple, Type

import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

import torch
import wandb
from scipy.interpolate import interp1d
from sklearn.metrics import (auc, average_precision_score,
                             precision_recall_curve, roc_auc_score, roc_curve)
from torch import Tensor
from torch.profiler import tensorboard_trace_handler

import data_dev01
import mvtec_dataset_dev01 as mvtec_dataset_dev01
from common_dev01 import create_python_random_generator, create_seed, seed_int2str, seed_str2int
from data_dev01 import ANOMALY_TARGET, NOMINAL_TARGET
import random


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



class RandomCallbackMixin:
    """
    Mixin that will get random generators from the init kwargs and make sure it is valid.
    """
    
    def init_python_generator(__init__):
        """Make sure that an arge `python_generator` is given as a kwarg at the init function."""
        
        @functools.wraps(__init__)
        def wrapper(self, *args, **kwargs):
            assert "python_generator" in kwargs, "key argument `python_generator` must be provided"
            gen: random.Random = kwargs.pop("python_generator")
            assert gen is not None, f"python_generator must not be None"
            assert isinstance(gen, random.Random), f"gen must be a random.Random, got {type(gen)}"
            self.python_generator = gen
            __init__(self, *args, **kwargs)
        
        return wrapper
    
    def init_torch_generator(__init__):
        """Make sure that an arge `torch_generator` is given as a kwarg at the init function."""
        
        @functools.wraps(__init__)
        def wrapper(self, *args, **kwargs):
            assert "torch_generator" in kwargs, "key argument `torch_generator` must be provided"
            gen: torch.Generator = kwargs.pop("torch_generator")
            assert gen is not None, f"torch_generator must not be None"
            assert isinstance(gen, torch.Generator), f"gen must be a torch.Generator, got {type(gen)}"
            self.torch_generator = gen
            __init__(self, *args, **kwargs)
        
        return wrapper

class LastEpochOutputsCallbackMixin:
    
    def setup_verify_last_epoch_outputs_is_none(pl_module_setup):
        """
        make sure that the attribute `last_epoch_outputs` is None at the beginning of a new stage
        otherwise it's because it hasn't been cleaned up properly
        """
        
        @functools.wraps(pl_module_setup)
        def wrapper(self, trainer, pl_module, stage=None):
            """self is the callback"""

            assert hasattr(pl_module, "last_epoch_outputs"), f"pl_module {type(pl_module)} must have a `last_epoch_outputs` attribute, got {type(pl_module)}"

            assert pl_module.last_epoch_outputs is None, f"pl_module.last_epoch_outputs must be None at the beginning of stage (={stage}), did you forget to clean up in the teardown()?  got {pl_module.last_epoch_outputs}"
        
            return pl_module_setup(self, trainer, pl_module, stage=stage)
        
        return wrapper

PYTORCH_LIGHTNING_STAGE_VALIDATE = "validate"
# PYTORCH_LIGHTNING_STAGE_SANITY_CHECK = "sanity_check"
PYTORCH_LIGHTNING_STAGE_TEST = "test"
PYTORCH_LIGHTNING_STAGE_TRAIN = "train"
PYTORCH_LIGHTNING_STAGE_PREDICT = "predict"
PYTORCH_LIGHTNING_STAGES = (
    PYTORCH_LIGHTNING_STAGE_VALIDATE,
    PYTORCH_LIGHTNING_STAGE_TEST,
    PYTORCH_LIGHTNING_STAGE_TRAIN,
    PYTORCH_LIGHTNING_STAGE_PREDICT,
)



def roc_curve(
    y_true=None, y_probas=None, labels=None, classes_to_plot=None, title=None
):
    """
    Calculates receiver operating characteristic scores and visualizes them as the
    ROC curve.
    
    i copied and adapted wandb.plot.roc_curve()
    i marked "modif" where i added something

    Arguments:
        y_true (arr): Test set labels.
        y_probas (arr): Test set predicted probabilities.
        labels (list): Named labels for target varible (y). Makes plots easier to
                        read by replacing target values with corresponding index.
                        For example labels= ['dog', 'cat', 'owl'] all 0s are
                        replaced by 'dog', 1s by 'cat'.

    Returns:
        Nothing. To see plots, go to your W&B run page then expand the 'media' tab
            under 'auto visualizations'.

    Example:
        ```
        wandb.log({'roc-curve': wandb.plot.roc_curve(y_true, y_probas, labels)})
        ```
    """
    import wandb
    from wandb import util
    from wandb.plots.utils import test_missing, test_types

    chart_limit = wandb.Table.MAX_ROWS
    
    np = util.get_module(
        "numpy",
        required="roc requires the numpy library, install with `pip install numpy`",
    )
    util.get_module(
        "sklearn",
        required="roc requires the scikit library, install with `pip install scikit-learn`",
    )
    from sklearn.metrics import roc_curve

    if test_missing(y_true=y_true, y_probas=y_probas) and test_types(
        y_true=y_true, y_probas=y_probas
    ):
        y_true = np.array(y_true)
        y_probas = np.array(y_probas)
        classes = np.unique(y_true)
        probas = y_probas
        
        # modif
        (nsamples, nscores) = y_probas.shape
        is_single_score = nscores == 1
        if is_single_score:
            assert tuple(classes) == (0, 1), "roc requires binary classification if there is a single score"
            assert classes_to_plot is None, f"classes_to_plot must be None if there is a single score"
            assert labels is None or len(labels) == 1, f"labels must be None or have length 1 if there is a single score"
        # modif
            
        if classes_to_plot is None:
            classes_to_plot = classes 

        fpr_dict = dict()
        tpr_dict = dict()

        indices_to_plot = np.in1d(classes, classes_to_plot)

        data = []
        count = 0
        
        # modif
        # very hacky but who cares
        if is_single_score:  # use only the positive score
            classes_to_plot = [classes_to_plot[1]]
            indices_to_plot = [indices_to_plot[1]] 
            classes = [classes[1]]
            probas = probas.reshape(-1, 1)
        # modif

        for i, to_plot in enumerate(indices_to_plot):
            
            # modif
            if is_single_score and i > 0:
                break
            # modif
            
            fpr_dict[i], tpr_dict[i], _ = roc_curve(
                y_true, probas[:, i], pos_label=classes[i]
            )
                
            if to_plot:
                for j in range(len(fpr_dict[i])):
                    if labels is not None and (
                        isinstance(classes[i], int)
                        or isinstance(classes[0], np.integer)
                    ):
                        class_dict = labels[classes[i]]
                    else:
                        class_dict = classes[i]
                    fpr = [
                        class_dict,
                        round(fpr_dict[i][j], 3),
                        round(tpr_dict[i][j], 3),
                    ]
                    data.append(fpr)
                    count += 1
                    if count >= chart_limit:
                        wandb.termwarn(
                            "wandb uses only the first %d datapoints to create the plots."
                            % wandb.Table.MAX_ROWS
                        )
                        break
        table = wandb.Table(columns=["class", "fpr", "tpr"], data=data)
        title = title or "ROC"
        return wandb.plot_table(
            "wandb/area-under-curve/v0",
            table,
            {"x": "fpr", "y": "tpr", "class": "class"},
            {
                "title": title,
                "x-axis-title": "False positive rate",
                "y-axis-title": "True positive rate",
            },
        )


class LogRocCurveCallback(
    LastEpochOutputsCallbackMixin, 
    RandomCallbackMixin, 
    pl.Callback,
):
    
    @RandomCallbackMixin.init_python_generator
    def __init__(
        self, 
        stage: str,
        scores_key: str,
        gt_key: str,
        log_curve: bool = True, 
        log_auc: bool = True, 
        limit_points: int = 3000,
    ):
        """
        Log the ROC curve (and it AUC) on val/test batches.
        
        Args:
            limit_points (int, optional): sample this number of points from all the available scores, if None, then no limitation is applied. Defaults to 3000.
        """
        super().__init__()
        
        assert stage in PYTORCH_LIGHTNING_STAGES, f"stage must be one of {PYTORCH_LIGHTNING_STAGES}, got {stage}"
        if stage not in (PYTORCH_LIGHTNING_STAGE_VALIDATE, PYTORCH_LIGHTNING_STAGE_TEST):
            raise NotImplementedError(f"ROC curve can only be logged on val/test, got {stage}")
        
        assert scores_key != "", f"scores_key must not be empty"
        assert gt_key != "", f"gt_key must not be empty"
        assert scores_key != gt_key, f"scores_key and gt_key must be different, got {scores_key} and {gt_key}"
        
        if limit_points is not None:
            assert limit_points > 0, f"limit_points must be > 0 or None, got {limit_points}"
        
        assert log_curve or log_auc, f"log_curve and log_auc must be True at least one of them"    

        self.stage = stage
        self.scores_key = scores_key
        self.gt_key = gt_key
        self.log_curve = log_curve
        self.log_auc = log_auc
        self.limit_points = limit_points
    
    @LastEpochOutputsCallbackMixin.setup_verify_last_epoch_outputs_is_none
    def setup(self, trainer, pl_module, stage=None):
        pass  # just let the mixin do its job
    
    def _log_roc_curve(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        
        try:
            scores = pl_module.last_epoch_outputs[self.scores_key]
            binary_gt = pl_module.last_epoch_outputs[self.gt_key]
        
        except KeyError as ex:
            msg = ex.args[0]
            if self.scores_key not in msg and self.gt_key not in msg:
                raise ex
            raise ArgumentError(f"pl_module.last_epoch_outputs should have the keys self.score_key={self.scores_key} and self.gt_key={self.gt_key}, did you configure the model correctly? or passed me the wrong keys?") from ex
        
        assert isinstance(scores, torch.Tensor), f"scores must be a torch.Tensor, got {type(scores)}"
        assert isinstance(binary_gt, torch.Tensor), f"binary_gt must be a torch.Tensor, got {type(binary_gt)}"
        
        assert scores.shape == binary_gt.shape, f"scores and binary_gt must have the same shape, got {scores.shape} and {binary_gt.shape}"
        
        unique_gt_values = tuple(sorted(torch.unique(binary_gt)))
        assert unique_gt_values in ((0,), (1,), (0, 1)), f"binary_gt must have only 0 and 1 values, got {unique_gt_values}"
        
        assert (scores >= 0).all(), f"scores must be >= 0, got {scores}"
        
        # make sure they are 1D (it doesn't matter if they were not before)
        scores = scores.reshape(-1, 1)  # wandb.plot.roc_curve() wants it like this
        binary_gt = binary_gt.reshape(-1)
        
        npoints = binary_gt.shape[0]
        
        if self.limit_points is not None and npoints > self.limit_points:
            indices = torch.tensor(self.python_generator.sample(range(npoints), self.limit_points))
            scores = scores[indices]
            binary_gt = binary_gt[indices]
        
        logkey_prefix = f"{trainer.state.stage}/" if trainer.state.stage is not None else ""
        curve_logkey = f"{logkey_prefix}roc-curve"
        auc_logkey = f"{logkey_prefix}roc-auc"
        # logdict = dict()
        
        if self.log_curve:
            # i copied and adapted wandb.plot.roc_curve()
            import wandb
            wandb.log({curve_logkey: roc_curve(binary_gt, scores, labels=["anomalous"])})
           
        if self.log_auc:
            trainer.model.log(auc_logkey, roc_auc_score(binary_gt, scores))
         
    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        
        if self.stage == PYTORCH_LIGHTNING_STAGE_VALIDATE and trainer.state.stage == PYTORCH_LIGHTNING_STAGE_VALIDATE: 
            self._log_roc_curve(trainer, pl_module)

    def on_test_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        
        if self.stage == PYTORCH_LIGHTNING_STAGE_TEST and trainer.state.stage == PYTORCH_LIGHTNING_STAGE_TEST: 
            self._log_roc_curve(trainer, pl_module)

        
class TorchTensorboardProfilerCallback(pl.Callback):
  """
  Quick-and-dirty Callback for invoking TensorboardProfiler during training.
  
  For greater robustness, extend the pl.profiler.profilers.BaseProfiler. See
  https://pytorch-lightning.readthedocs.io/en/stable/advanced/profiler.html
  """

  def __init__(self, profiler):
    super().__init__()
    self.profiler = profiler 

  def on_train_batch_end(self, trainer, pl_module, outputs, *args, **kwargs):
    self.profiler.step()
    # pl_module.log_dict(outputs)  # also logging the loss, while we're here


class DataloaderPreviewCallback(pl.Callback):
    
    def __init__(self, dataloader, n_samples=5,  logkey_prefix="train/preview"):
        """the keys loggeda are `{logkey_prefix}/anomalous` and `{logkey_prefix}/normal`"""
        super().__init__()
        assert isinstance(dataloader, torch.utils.data.DataLoader), f"dataloader must be a torch.utils.data.DataLoader, got {type(dataloader)}"
        assert n_samples > 0, f"n_samples must be > 0, got {n_samples}"
        self.dataloader = dataloader
        self.n_samples = n_samples
        self.logkey_prefix = logkey_prefix
        
    @staticmethod
    def _get_mask_dict(mask):
        """mask_tensor \in int32^[1, H, W]"""
        return dict(
            ground_truth=dict(
                mask_data=mask.squeeze().numpy(), 
                class_labels={
                    NOMINAL_TARGET: "normal", 
                    ANOMALY_TARGET: "anomalous",
                }
            )
        )

    def on_fit_start(self, trainer, model):
        (
            norm_imgs, norm_gtmaps, anom_imgs, anom_gtmaps
        ) = data_dev01.generate_dataloader_images(self.dataloader, nimages_perclass=self.n_samples)

        import wandb
        wandb.log({
            f"{self.logkey_prefix}/normal": [
                wandb.Image(img, caption=[f"normal {idx:03d}"], masks=self._get_mask_dict(mask))
                for idx, (img, mask) in enumerate(zip(norm_imgs, norm_gtmaps))
            ],
            f"{self.logkey_prefix}/anomalous": [
                wandb.Image(img, caption=[f"anomalous {idx:03d}"], masks=self._get_mask_dict(mask))
                for idx, (img, mask) in enumerate(zip(anom_imgs, anom_gtmaps))
            ],
        })

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
        help="Argument for wandb_logger.watch(..., log_freq=WANDB_WATCH_LOG_FREQ).",
    )
    parser.add_argument(        
        "--wandb-checkpoint-mode", type=str, choices=WANDB_CHECKPOINT_MODES, default=WANDB_CHECKPOINT_MODE_LAST,
        help="How to save checkpoints."
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
# ========================================== run ===========================================
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
    # pytorch lightning
    lightning_accelerator: str,
    lightning_ndevices: int,
    lightning_strategy: str,
    lightning_precision: str,
    lightning_model_summary_max_depth: int,
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
        LogRocCurveCallback(
            stage=PYTORCH_LIGHTNING_STAGE_VALIDATE,
            scores_key="anomaly_scores_maps",
            gt_key="gtmaps",
            log_curve=True,
            log_auc=True,
            # test_steps_outputs_parser=model.test_step_outputs_get_scores_and_gt_maps,
            limit_points=3000,  # todo make it script param
            python_generator=create_python_random_generator(seed),
        )
    ]
    
    if preview_nimages > 0:
        datamodule.setup("fit")
        callbacks.append(
            DataloaderPreviewCallback(
                dataloader=datamodule.train_dataloader(embed_preprocessing=True), 
                n_samples=preview_nimages, logkey_prefix="train-preview"
            ),
        )
    
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
        # todo make specific callbacks on LightningModule.configure_callbacks()
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
                if "got an unexpected keyword argument" in msg:
                    raise ScriptError(f"run_one() got an unexpected keyword argument: {msg}, did you forget to kwargs.pop() something?") from ex
                raise ex
            except Exception as ex:
                wandb_logger.finalize("failed")
                wandb.finish(1)
                raise ex
            else:
                wandb_logger.finalize("success")
                wandb.finish(0)
                