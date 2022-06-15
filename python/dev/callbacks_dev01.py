#!/usr/bin/env python
# coding: utf-8

import abc
import functools
import random
from tkinter import W
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.trainer.states import RunningStage
from sklearn.metrics import (roc_auc_score, average_precision_score)
from torch import Tensor

from data_dev01 import ANOMALY_TARGET, NOMINAL_TARGET


def merge_steps_outputs(steps_outputs: List[Dict[str, Tensor]]):
    """
    gather all the pieces into a single dict and transfer tensors to cpu
    
    several callbacks depend on the pl_module (model) having the attribute last_epoch_outputs,
    which should be set with this function
    """
    
    if steps_outputs is None:
        return None
    
    assert isinstance(steps_outputs, list), f"steps_outputs must be a list, got {type(steps_outputs)}"

    assert len(steps_outputs) >= 1, f"steps_outputs must have at least one element, got {len(steps_outputs)}"

    assert all(isinstance(x, dict) for x in steps_outputs), f"steps_outputs must be a list of dicts, got {steps_outputs}"

    dict0 = steps_outputs[0]
    keys = set(dict0.keys())

    assert all(set(x.keys()) == keys for x in steps_outputs), f"steps_outputs must have the same keys, got {steps_outputs}, keys={keys}"
    
    return {
        # step 3: transfer the tensor to the cpu to liberate the gpu memory
        k: v.cpu() 
        for k, v in {
            # step 2: concatenate the tensors
            k: (
                torch.stack(list_of_values, dim=0)
                if list_of_values[0].ndim == 0 else
                torch.cat(list_of_values, dim=0)
            )
            for k, list_of_values in {
                # step 1: gather the dicts values into lists (one list per key) ==> single dict
                k: [step_outputs[k] for step_outputs in steps_outputs]
                for k in keys
            }.items()
        }.items()
    }
    

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


class LastEpochOutputsDependentCallbackMixin:
    
    def setup_verify_plmodule_with_last_epoch_outputs(pl_module_setup):
        """
        this should be used as a decorator on top of setup() of callbacks that depend on last_epoch_outputs
        
        make sure that the attribute `last_epoch_outputs` is None at the beginning of a new stage
        otherwise it's because it hasn't been cleaned up properly
        """
        
        @functools.wraps(pl_module_setup)
        def wrapper(self, trainer, pl_module, stage=None):
            """self is the callback"""
            
            assert hasattr(pl_module, "last_epoch_outputs"), f"{self} must have attribute `last_epoch_outputs`"

            assert pl_module.last_epoch_outputs is None, f"pl_module.last_epoch_outputs must be None at the beginning of stage (={stage}), did you forget to clean up in the teardown()?  got {pl_module.last_epoch_outputs}"
        
            return pl_module_setup(self, trainer, pl_module, stage=stage)
        
        return wrapper


class MultiStageEpochEndCallbackMixin(abc.ABC):
    """
    use this mixin to call a callback on multiple stages
    you must decorate __init__ with init_stage_arg()
    when using this mixin, you should define the function _multi_stage_epoch_end_do() in your module
    """
    
    ACCEPTED_STAGES = [
        RunningStage.TRAINING,
        RunningStage.VALIDATING,
        RunningStage.TESTING,
    ]
    
    @abc.abstractmethod
    def _multi_stage_epoch_end_do(self, *args, **kwargs):
        pass
    
    def init_stage(__init__):
        """Make sure that an arg `stage` is given as a kwarg at the init function and is valid"""
        
        @functools.wraps(__init__)
        def wrapper(self, *args, **kwargs):            
            assert "stage" in kwargs, "key argument `stage` must be provided"
            stage: Union[str, RunningStage] = kwargs.pop("stage")
            assert stage is not None, f"stage must not be None"
            assert stage in MultiStageEpochEndCallbackMixin.ACCEPTED_STAGES, f"stage must be one of {MultiStageEpochEndCallbackMixin.ACCEPTED_STAGES}, got {stage}"
            self.stage = stage
            __init__(self, *args, **kwargs)
        
        return wrapper
    
    @staticmethod
    def should_log(hook_stage, callback_stage, trainer_stage, ):
        return callback_stage == hook_stage and trainer_stage == hook_stage
    
    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        if self.should_log(RunningStage.TRAINING, self.stage, trainer.state.stage) and pl_module.last_epoch_outputs is not None:
            self._multi_stage_epoch_end_do(trainer, pl_module)
    
    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        if self.should_log(RunningStage.VALIDATING, self.stage, trainer.state.stage) and pl_module.last_epoch_outputs is not None: 
            self._multi_stage_epoch_end_do(trainer, pl_module)

    def on_test_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        if self.should_log(RunningStage.TESTING, self.stage, trainer.state.stage) and pl_module.last_epoch_outputs is not None: 
            self._multi_stage_epoch_end_do(trainer, pl_module)
            
            
class LogRocCallback(
    MultiStageEpochEndCallbackMixin,
    LastEpochOutputsDependentCallbackMixin, 
    RandomCallbackMixin, 
    pl.Callback,
):
    
    @MultiStageEpochEndCallbackMixin.init_stage  
    @RandomCallbackMixin.init_python_generator
    def __init__(
        self, 
        scores_key: str,
        gt_key: str,
        log_curve: bool = False, 
        limit_points: int = 3000,
    ):
        """
        Log the area under the roc curve and (optionally) the curve itself at the end of an epoch.
        
        Args:
            limit_points (int, optional): sample this number of points from all the available scores, if None, then no limitation is applied. Defaults to 3000.
        """
        super().__init__()
                
        assert scores_key != "", f"scores_key must not be empty"
        assert gt_key != "", f"gt_key must not be empty"
        assert scores_key != gt_key, f"scores_key and gt_key must be different, got {scores_key} and {gt_key}"
        
        if limit_points is not None:
            assert limit_points > 0, f"limit_points must be > 0 or None, got {limit_points}"
            
        self.scores_key = scores_key
        self.gt_key = gt_key
        self.log_curve = log_curve
        self.limit_points = limit_points
    
    @LastEpochOutputsDependentCallbackMixin.setup_verify_plmodule_with_last_epoch_outputs
    def setup(self, trainer, pl_module, stage=None):
        pass  # just let the mixin do its job
    
    def _multi_stage_epoch_end_do(self, trainer, pl_module):
        if pl_module.last_epoch_outputs is None:
            return
        self._log_roc(trainer, pl_module)
    
    def _log_roc(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        
        try:
            scores = pl_module.last_epoch_outputs[self.scores_key]
            binary_gt = pl_module.last_epoch_outputs[self.gt_key]
        
        except KeyError as ex:
            msg = ex.args[0]
            if self.scores_key not in msg and self.gt_key not in msg:
                raise ex
            raise ValueError(f"pl_module.last_epoch_outputs should have the keys self.score_key={self.scores_key} and self.gt_key={self.gt_key}, did you configure the model correctly? or passed me the wrong keys?") from ex
        
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
        
        if self.log_curve:
            import wandb
            # i copied and adapted wandb.plot.roc_curve.roc_curve()
            import hacked_dev01
            wandb.log({curve_logkey: hacked_dev01.roc_curve(binary_gt, scores, labels=["anomalous"])})
           
        trainer.model.log(auc_logkey, roc_auc_score(binary_gt, scores))


class LogAveragePrecisionCallback(
    MultiStageEpochEndCallbackMixin,
    LastEpochOutputsDependentCallbackMixin, 
    RandomCallbackMixin, 
    pl.Callback,
):
    
    @MultiStageEpochEndCallbackMixin.init_stage  
    @RandomCallbackMixin.init_python_generator
    def __init__(
        self, 
        scores_key: str,
        gt_key: str,
        log_curve: bool = False, 
        limit_points: int = 3000,
    ):
        """
        Log the average precision and (optionally) its curve (precision-recall curve) at the end of an epoch.
        
        Args:
            scores_key & gt_key: inside the last_epoch_outputs, how are the scores and gt maps called?
            limit_points (int, optional): sample this number of points from all the available scores, if None, then no limitation is applied. Defaults to 3000.
        """
        super().__init__()
                
        assert scores_key != "", f"scores_key must not be empty"
        assert gt_key != "", f"gt_key must not be empty"
        assert scores_key != gt_key, f"scores_key and gt_key must be different, got {scores_key} and {gt_key}"
        
        if limit_points is not None:
            assert limit_points > 0, f"limit_points must be > 0 or None, got {limit_points}"
            
        self.scores_key = scores_key
        self.gt_key = gt_key
        self.log_curve = log_curve
        self.limit_points = limit_points
    
    @LastEpochOutputsDependentCallbackMixin.setup_verify_plmodule_with_last_epoch_outputs
    def setup(self, trainer, pl_module, stage=None):
        pass  # just let the mixin do its job
    
    def _multi_stage_epoch_end_do(self, trainer, pl_module):
        self._log_avg_precision(trainer, pl_module)
    
    def _log_avg_precision(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        
        try:
            scores = pl_module.last_epoch_outputs[self.scores_key]
            binary_gt = pl_module.last_epoch_outputs[self.gt_key]
        
        except KeyError as ex:
            msg = ex.args[0]
            if self.scores_key not in msg and self.gt_key not in msg:
                raise ex
            raise ValueError(f"pl_module.last_epoch_outputs should have the keys self.score_key={self.scores_key} and self.gt_key={self.gt_key}, did you configure the model correctly? or passed me the wrong keys?") from ex
        
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
        
        current_stage = trainer.state.stage
        logkey_prefix = f"{current_stage}/" if current_stage is not None else ""
        curve_logkey = f"{logkey_prefix}precision-recall-curve"
        avg_precision_logkey = f"{logkey_prefix}avg-precision"
        
        if self.log_curve:
            import wandb
            # i copied and adapted wandb.plot.pr_curve.pr_curve()
            import hacked_dev01
            wandb.log({curve_logkey: hacked_dev01.pr_curve(binary_gt, scores, labels=["anomalous"])})
           
        trainer.model.log(avg_precision_logkey, average_precision_score(binary_gt, scores))
  
  
LOG_HISTOGRAM_MODE_NONE = "none"
LOG_HISTOGRAM_MODE_LOG = "log"
LOG_HISTOGRAM_MODE_SUMMARY = "summary"
LOG_HISTOGRAM_MODES = (LOG_HISTOGRAM_MODE_NONE, LOG_HISTOGRAM_MODE_LOG, LOG_HISTOGRAM_MODE_SUMMARY)
       
class LogHistogramCallback(
    MultiStageEpochEndCallbackMixin,
    LastEpochOutputsDependentCallbackMixin, 
    pl.Callback,
):
    
    @MultiStageEpochEndCallbackMixin.init_stage  
    def __init__(self, key: str, mode: str):
        """
        Args:
            key: inside the last_epoch_outputs, what do you want to log?
        """
        super().__init__()
        assert key != "", f"key must not be empty"
        assert mode in LOG_HISTOGRAM_MODES, f"log_mode must be one of {LOG_HISTOGRAM_MODES}, got '{mode}'"
        if mode == LOG_HISTOGRAM_MODE_NONE:
            raise ValueError(f"mode={mode} should not be used, just dont add the callback to the trainer :)")
        self.key = key
        self.mode = mode

    @LastEpochOutputsDependentCallbackMixin.setup_verify_plmodule_with_last_epoch_outputs
    def setup(self, trainer, pl_module, stage=None):
        pass  # just let the mixin do its job
        
    def _multi_stage_epoch_end_do(self, trainer, pl_module):
        self._log_histogram(trainer, pl_module)
    
    def _log_histogram(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        
        try:
            values = pl_module.last_epoch_outputs[self.key]
        
        except KeyError as ex:
            msg = ex.args[0]
            if self.key not in msg:
                raise ex
            raise ValueError(f"pl_module.last_epoch_outputs should have the key self.key={self.key}, did you configure the model correctly? or passed me the wrong key?") from ex
        
        assert isinstance(values, torch.Tensor), f"scores must be a torch.Tensor, got {type(values)}"
        
        current_stage = trainer.state.stage
        logkey_prefix = f"{current_stage}/" if current_stage is not None else ""
        logkey = f"{logkey_prefix}histogram-of-{self.key}"
        
        # requirement from wandb.Histogram()
        values = values.detach().numpy() if values.requires_grad else values
        
        import wandb
        
        if self.mode == LOG_HISTOGRAM_MODE_LOG:
            wandb.log({logkey: wandb.Histogram(values)})
            
        elif self.mode == LOG_HISTOGRAM_MODE_SUMMARY:
            wandb.run.summary.update({logkey: wandb.Histogram(values)})
            
        else:
            raise ValueError(f"log_mode must be one of {LOG_HISTOGRAM_MODES}, got '{self.mode}'")
        
        
class LogHistogramsSuperposedPerClassCallback(
    MultiStageEpochEndCallbackMixin,
    LastEpochOutputsDependentCallbackMixin, 
    RandomCallbackMixin, 
    pl.Callback,
):
    
    @MultiStageEpochEndCallbackMixin.init_stage
    @RandomCallbackMixin.init_python_generator
    def __init__(self, values_key: str, gt_key: str, mode: str, limit_points: int = 3000,):
        """
        Args:
            values_key & gt_key: inside the last_epoch_outputs, how are the values and their respective labels called?
        """
        super().__init__()
                
        assert values_key != "", f"scores_key must not be empty"
        assert gt_key != "", f"gt_key must not be empty"
        assert values_key != gt_key, f"scores_key and gt_key must be different, got {values_key} and {gt_key}"
        
        assert mode in LOG_HISTOGRAM_MODES, f"log_mode must be one of {LOG_HISTOGRAM_MODES}, got '{mode}'"
        if mode == LOG_HISTOGRAM_MODE_NONE:
            raise ValueError(f"mode={mode} should not be used, just dont add the callback to the trainer :)")
                
        if limit_points is not None:
            assert limit_points > 0, f"limit_points must be > 0 or None, got {limit_points}"
            
        self.values_key = values_key
        self.gt_key = gt_key
        self.mode = mode
        self.limit_points = limit_points

    @LastEpochOutputsDependentCallbackMixin.setup_verify_plmodule_with_last_epoch_outputs
    def setup(self, trainer, pl_module, stage=None):
        pass  # just let the mixin do its job
        
    def _multi_stage_epoch_end_do(self, trainer, pl_module):
        self._log_histograms_supperposed_per_class(trainer, pl_module)
    
    def _log_histograms_supperposed_per_class(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        
        try:
            values = pl_module.last_epoch_outputs[self.values_key]
            gt = pl_module.last_epoch_outputs[self.gt_key]
            
        except KeyError as ex:
            msg = ex.args[0]
            if self.values_key not in msg and self.gt_key not in msg:
                raise ex
            raise ValueError(f"pl_module.last_epoch_outputs should have the keys self.values_key={self.values_key} and self.gt_key={self.gt_key}, did you configure the model correctly? or passed me the wrong keys?") from ex
        
        assert isinstance(values, torch.Tensor), f"values must be a torch.Tensor, got {type(values)}"
        assert isinstance(gt, torch.Tensor), f"gt must be a torch.Tensor, got {type(gt)}"

        assert values.shape == gt.shape, f"values and gtg must have the same shape, got {values.shape} and {gt.shape}"
                        
        # make sure they are 1D (it doesn't matter if they were not before)
        values = values.reshape(-1, 1)  # wandb.plot.roc_curve() wants it like this
        gt = gt.reshape(-1, 1)
                
        if self.limit_points is None:
            values_selected = values
            gt_selected = gt
            
        else:
            unique_gt_values = gt.unique()
            n_unique_gt_values = len(unique_gt_values)
            
            values_selected = torch.empty(self.limit_points * n_unique_gt_values, 1, dtype=values.dtype)
            gt_selected = torch.empty(self.limit_points * n_unique_gt_values, 1, dtype=gt.dtype)
            
            for idx, gt_value in enumerate(unique_gt_values): 
                
                from_idx = idx * self.limit_points
                to_idx = (idx + 1) * self.limit_points
                
                values_ = values[gt == gt_value].unsqueeze(-1)
                npoints_ = values_.shape[0]

                if npoints_ > self.limit_points:
                    indices = torch.tensor(self.python_generator.sample(range(npoints_), self.limit_points))
                    
                    values_selected[from_idx:to_idx] = values_[indices]
                    gt_selected[from_idx:to_idx] = torch.full((self.limit_points, 1), gt_value, dtype=gt.dtype)
                    
                else:
                    # if npoints < self.limit_points, there will be empty things there
                    # so i'll put nans and later remove them
                    tmp = torch.empty(self.limit_points, 1, dtype=values.dtype)
                    tmp[:npoints_] = values_
                    tmp[npoints_:] = np.nan
                    values_selected[from_idx:to_idx] = tmp
                    gt_selected[from_idx:to_idx] = torch.full((self.limit_points, 1), gt_value, dtype=gt.dtype)
            
            values_to_keep = ~ torch.isnan(values_selected)
            values_selected = values_selected[values_to_keep].unsqueeze(-1)
            gt_selected = gt_selected[values_to_keep].unsqueeze(-1)
        
        table = torch.cat((values_selected, gt_selected), dim=1).tolist()
        
        current_stage = trainer.state.stage
        logkey_prefix = f"{current_stage}/" if current_stage is not None else ""
        logkey = f"{logkey_prefix}histogram-superposed-per-class-of-{self.values_key}"
             
        import wandb
        
        table = wandb.Table(data=table, columns=[self.values_key, "gt"])
        
        if self.mode == LOG_HISTOGRAM_MODE_LOG:
            wandb.log({logkey: table})
            
        elif self.mode == LOG_HISTOGRAM_MODE_SUMMARY:
            raise NotImplementedError(f"mode {self.mode} was having display problems in wandb")
            wandb.run.summary.update({logkey: table})
            
        else:
            raise ValueError(f"mode must be one of {LOG_HISTOGRAM_MODES}, got '{self.mode}'")      
        
        
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
        import data_dev01
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
