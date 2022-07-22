#!/usr/bin/env python
# coding: utf-8

import abc
import functools
import random
import re
from this import d
import warnings
from argparse import ArgumentParser
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torchvision.transforms.functional_tensor as TFT
from pytorch_lightning.trainer.states import RunningStage
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import Tensor

import data_dev01_bis
import hacked_dev01
import wandb
from common_dev01_bis import (AdaptiveClipError, ArgumentParserOrArgumentGroup,
                              RunningStageOrStr,
                              create_python_random_generator,
                              find_scores_clip_values_from_empircal_cdf,
                              none_or_int)
from data_dev01_bis import ANOMALY_TARGET, NOMINAL_TARGET

from pytorch_lightning.loggers import WandbLogger

CliArgNameMap = Dict[str, str]


LOG_HISTOGRAM_MODE_LOG = "log"
LOG_HISTOGRAM_MODE_SUMMARY = "summary"
LOG_HISTOGRAM_MODES = (None, LOG_HISTOGRAM_MODE_LOG, LOG_HISTOGRAM_MODE_SUMMARY)

HEATMAP_NORMALIZATION_MINMAX_IN_EPOCH = "minmax-epoch"
HEATMAP_NORMALIZATION_MINMAX_INSTANCE = "minmax-instance"
HEATMAP_NORMALIZATION_PERCENTILES_IN_EPOCH = "percentiles-epoch"
HEATMAP_NORMALIZATION_PERCENTILES_ADAPTIVE_CDF_BASED_IN_EPOCH = "percentiles-adaptive-cdf-epoch"
HEATMAP_NORMALIZATION_CHOICES = (
    HEATMAP_NORMALIZATION_PERCENTILES_IN_EPOCH,
    HEATMAP_NORMALIZATION_MINMAX_IN_EPOCH,
    HEATMAP_NORMALIZATION_MINMAX_INSTANCE,
    HEATMAP_NORMALIZATION_PERCENTILES_ADAPTIVE_CDF_BASED_IN_EPOCH,
)

REGEXSTR_ASSERT_VARIABLE_NAME = "^[a-zA-Z_][a-zA-Z0-9_]*$"


class RandomCallbackMixin:
    """
    Mixin that will get random generators from the init kwargs and make sure it is valid.
    """
    
    @property
    def python_generator(self) -> random.Random:
        
        if not hasattr(self, "_python_generator"):
            raise AttributeError("python_generator must be set first (did you miss something in the callback init?)")

        return self._python_generator
    
    @python_generator.setter
    def python_generator(self, value: random.Random):
        
        if hasattr(self, "_python_generator"):
            raise AttributeError("python_generator already set, what are you doing?")
    
        assert value is not None, f"python_generator must not be None"
        assert isinstance(value, random.Random), f"python_generator must be a random.Random, got {type(value)}"
    
        self._python_generator = value


class LastEpochOutputsDependentCallbackMixin:

    def setup_validate_plmodule_last_epoch_outputs(self, trainer, pl_module, stage=None):
        """
        this should be called in setup() of callbacks that depend on last_epoch_outputs

        make sure that the attribute `last_epoch_outputs` is None at the beginning of a new stage
        otherwise it's because it hasn't been cleaned up properly
        """
        
        assert hasattr(pl_module, "last_epoch_outputs"), f"{pl_module.__class__.__name__} must have attribute `last_epoch_outputs`"

        assert pl_module.last_epoch_outputs is None, f"pl_module.last_epoch_outputs must be None at the beginning of stage (={stage}), did you forget to clean up in the teardown()?  got {pl_module.last_epoch_outputs}"
        

class MultiStageCallbackMixin(abc.ABC):
    """
    stage multiplexer for callbacks that use on_{stage}_epoch_end 
    use this mixin to call a callback on one out of the accepted stages so you can use the same callback for multiple stages with different instances
    you must set the property `stage` to the stage at the __init__ of the callback
    when using this mixin, you should define the function one of the functions 
     - _multi_stage_epoch_end_do()
     - _multi_stage_end_do() 
    in your callback
    """

    ACCEPTED_STAGES = [
        RunningStage.TRAINING,
        RunningStage.VALIDATING,
        RunningStage.TESTING,
    ]

    def _multi_stage_epoch_end_do(self, *args, **kwargs):
        pass
    
    def _multi_stage_end_do(self, *args, **kwargs):
        pass
    
    @property
    def stage(self) -> RunningStage:
        
        if not hasattr(self, "_stage"):
            raise AttributeError("stage must be set first (did you miss something in the callback init?)")
        
        return self._stage
    
    @stage.setter
    def stage(self, value: RunningStage):
        
        if hasattr(self, "_stage"):
            raise AttributeError("stage already set, what are you doing?")
        
        assert isinstance(value, RunningStage) or isinstance(value, str), f"stage must be a RunningStage or a string, got {type(value)}"
        assert value in self.ACCEPTED_STAGES, f"stage must be one of {self.ACCEPTED_STAGES}, got {value}"
        
        self._stage = value

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

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        if self.should_log(RunningStage.TRAINING, self.stage, trainer.state.stage) and pl_module.last_epoch_outputs is not None:
            self._multi_stage_end_do(trainer, pl_module)

    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        if self.should_log(RunningStage.VALIDATING, self.stage, trainer.state.stage) and pl_module.last_epoch_outputs is not None:
            self._multi_stage_end_do(trainer, pl_module)

    def on_test_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        if self.should_log(RunningStage.TESTING, self.stage, trainer.state.stage) and pl_module.last_epoch_outputs is not None:
            self._multi_stage_end_do(trainer, pl_module)


class LogRocCallback(
    MultiStageCallbackMixin,
    LastEpochOutputsDependentCallbackMixin,
    RandomCallbackMixin,
    pl.Callback,
):
        
    @staticmethod
    def cli_add_arguments(parser: ArgumentParserOrArgumentGroup, stage: str, **defaults) -> CliArgNameMap:
        
        assert stage in MultiStageCallbackMixin.ACCEPTED_STAGES, f"stage must be one of {MultiStageCallbackMixin.ACCEPTED_STAGES}, got {stage}"
        assert isinstance(stage, str), f"stage must be a str, got {type(stage)}"
        
        parser.description = f"Log ROC and ROC-AUC score for stage={stage}."
        
        cli_arg_name_map = dict()
                
        def cliname(argname):
            cliname = f"logroc_{stage}_{argname}"
            cli_arg_name_map[cliname] = argname
            return cliname
        
        parser.add_argument(
            f'--{cliname("scores_key")}', 
            type=str, 
        )
        
        parser.add_argument(
            f'--{cliname("gt_key")}', 
            type=str, 
        )
        
        parser.add_argument(
            f'--{cliname("log_curve")}', 
            type=bool, 
        )
        
        parser.add_argument(
            f'--{cliname("limit_points")}', 
            type=int, 
        )
                
        # these shouldnt be changed
        parser.add_argument(f'--{cliname("stage")}', type=str, default=stage,)
        # the seed here can be fixed, no need to follow the seed of the other
        # random stuff because in anycase the training is already random, and 
        # this randomness is just for the sampling of points used to estimate the ROC
        # it's not a big deal if all trainings use the same seed
        parser.add_argument(f'--{cliname("python_generator")}', type=create_python_random_generator, default=create_python_random_generator(0),)
        parser.set_defaults(**{cliname(argname): argval for argname, argval in defaults.items()})
        
        return cli_arg_name_map
        
    def __init__(
        self,
        scores_key: str,
        gt_key: str,
        log_curve: bool,
        limit_points: int,
        # mixin args
        python_generator: random.Random,
        stage: RunningStageOrStr,
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
        
        # from mixins (validations already done in the mixin)
        self.python_generator = python_generator
        self.stage = stage
        
    def setup(self, trainer, pl_module, stage=None):
        self.setup_validate_plmodule_last_epoch_outputs(trainer, pl_module, stage=stage)

    def _multi_stage_epoch_end_do(self, trainer, pl_module):
        if pl_module.last_epoch_outputs is None:
            return
        self._log_roc(trainer, pl_module)

    def _log_roc(self, trainer: pl.Trainer, pl_module: pl.LightningModule):

        try:
            scores = pl_module.last_epoch_outputs[self.scores_key].detach()
            binary_gt = pl_module.last_epoch_outputs[self.gt_key]

        except KeyError as ex:
            msg = ex.args[0]
            if self.scores_key not in msg and self.gt_key not in msg:
                raise ex
            raise ValueError(f"pl_module.last_epoch_outputs should have the keys self.score_key={self.scores_key} and self.gt_key={self.gt_key}, did you configure the model correctly? or passed me the wrong keys?") from ex

        assert isinstance(scores, Tensor), f"scores must be a torch.Tensor, got {type(scores)}"
        assert isinstance(binary_gt, Tensor), f"binary_gt must be a torch.Tensor, got {type(binary_gt)}"

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
            # i copied and adapted wandb.plot.roc_curve.roc_curve()
            wandb.log({curve_logkey: hacked_dev01.roc_curve(binary_gt, scores, labels=["anomalous"])})

        trainer.model.log(auc_logkey, roc_auc_score(binary_gt, scores))


class LogPrcurveCallback(
    MultiStageCallbackMixin,
    LastEpochOutputsDependentCallbackMixin,
    RandomCallbackMixin,
    pl.Callback,
):
  
    @staticmethod
    def cli_add_arguments(parser: ArgumentParser, stage: str, **defaults) -> CliArgNameMap:
        
        assert stage in MultiStageCallbackMixin.ACCEPTED_STAGES, f"stage must be one of {MultiStageCallbackMixin.ACCEPTED_STAGES}, got {stage}"
        assert isinstance(stage, str), f"stage must be a str, got {type(stage)}"
        
        parser.description = f"Log PR curve for stage={stage}."
        
        cli_arg_name_map = dict()
                
        def cliname(argname):
            cliname = f"prcurve_{stage}_{argname}"
            cli_arg_name_map[cliname] = argname
            return cliname
        
        parser.add_argument(
            f'--{cliname("scores_key")}', 
            type=str, 
        )
        
        parser.add_argument(
            f'--{cliname("gt_key")}', 
            type=str, 
        )
        
        parser.add_argument(
            f'--{cliname("log_curve")}', 
            type=bool, 
        )
        
        parser.add_argument(
            f'--{cliname("limit_points")}', 
            type=int, 
        )
                
        # these shouldnt be changed
        parser.add_argument(f'--{cliname("stage")}', type=str, default=stage,)
        # the seed here can be fixed, no need to follow the seed of the other
        # random stuff because in anycase the training is already random, and 
        # this randomness is just for the sampling of points used to estimate the ROC
        # it's not a big deal if all trainings use the same seed
        parser.add_argument(f'--{cliname("python_generator")}', type=create_python_random_generator, default=create_python_random_generator(0),)
        parser.set_defaults(**{cliname(argname): argval for argname, argval in defaults.items()})
        
        return cli_arg_name_map
        
    def __init__(
        self,
        scores_key: str,
        gt_key: str,
        log_curve: bool,
        limit_points: int,
        # mixin args
        python_generator: random.Random,
        stage: RunningStageOrStr,
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
        
        # mixin args
        self.python_generator = python_generator
        self.stage = stage

    def setup(self, trainer, pl_module, stage=None):
        self.setup_validate_plmodule_last_epoch_outputs(trainer, pl_module, stage=stage)
        
    def _multi_stage_epoch_end_do(self, trainer, pl_module):
        self._log_avg_precision(trainer, pl_module)

    def _log_avg_precision(self, trainer: pl.Trainer, pl_module: pl.LightningModule):

        try:
            scores = pl_module.last_epoch_outputs[self.scores_key].detach()
            binary_gt = pl_module.last_epoch_outputs[self.gt_key]

        except KeyError as ex:
            msg = ex.args[0]
            if self.scores_key not in msg and self.gt_key not in msg:
                raise ex
            raise ValueError(f"pl_module.last_epoch_outputs should have the keys self.score_key={self.scores_key} and self.gt_key={self.gt_key}, did you configure the model correctly? or passed me the wrong keys?") from ex

        assert isinstance(scores, Tensor), f"scores must be a torch.Tensor, got {type(scores)}"
        assert isinstance(binary_gt, Tensor), f"binary_gt must be a torch.Tensor, got {type(binary_gt)}"

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
            # i copied and adapted wandb.plot.pr_curve.pr_curve()
            wandb.log({curve_logkey: hacked_dev01.pr_curve(binary_gt, scores, labels=["anomalous"])})

        trainer.model.log(avg_precision_logkey, average_precision_score(binary_gt, scores))


class LogHistogramCallback(
    MultiStageCallbackMixin,
    LastEpochOutputsDependentCallbackMixin,
    pl.Callback,
):

    @staticmethod
    def cli_add_arguments(parser: ArgumentParser, histogram_of: str, stage: str, **defaults) -> CliArgNameMap:
        
        assert stage in MultiStageCallbackMixin.ACCEPTED_STAGES, f"stage must be one of {MultiStageCallbackMixin.ACCEPTED_STAGES}, got {stage}"
        assert isinstance(stage, str), f"stage must be a str, got {type(stage)}"
        assert isinstance(histogram_of, str), f"histogram_of must be a str, got {type(histogram_of)}"
        assert re.match(REGEXSTR_ASSERT_VARIABLE_NAME, histogram_of), f"histogram_of must be a valid variable name, got {histogram_of}"
        
        parser.description = f"Log histogram of {histogram_of} for stage={stage}."
        
        cli_arg_name_map = dict()
                
        def cliname(argname):
            cliname = f"log_histogram_{histogram_of}_{stage}_{argname}"
            cli_arg_name_map[cliname] = argname
            return cliname
        
        parser.add_argument(
            f'--{cliname("key")}', 
            type=str, 
        )
        
        parser.add_argument(
            f'--{cliname("mode")}', 
            type=str, choices=LOG_HISTOGRAM_MODES,
        )
                
        # these shouldnt be changed
        parser.add_argument(f'--{cliname("stage")}', type=str, default=stage,)
        parser.set_defaults(**{cliname(argname): argval for argname, argval in defaults.items()})
        
        return cli_arg_name_map
       
    def __init__(self, key: str, mode: str, stage: RunningStageOrStr,):
        """
        Args:
            key: inside the last_epoch_outputs, what do you want to log?
        """
        super().__init__()
        assert key != "", f"key must not be empty"
        assert mode in LOG_HISTOGRAM_MODES, f"log_mode must be one of {LOG_HISTOGRAM_MODES}, got '{mode}'"
        assert mode is not None, f"log_mode must not be None, just dont add the callback to the trainer :)"
        self.key = key
        self.mode = mode
        
        # mixin args
        self.stage = stage

    def setup(self, trainer, pl_module, stage=None):
        self.setup_validate_plmodule_last_epoch_outputs(trainer, pl_module, stage=stage)

    def _multi_stage_epoch_end_do(self, trainer, pl_module):
        self._log_histogram(trainer, pl_module)

    def _log_histogram(self, trainer: pl.Trainer, pl_module: pl.LightningModule):

        try:
            values = pl_module.last_epoch_outputs[self.key].detach()

        except KeyError as ex:
            msg = ex.args[0]
            if self.key not in msg:
                raise ex
            raise ValueError(f"pl_module.last_epoch_outputs should have the key self.key={self.key}, did you configure the model correctly? or passed me the wrong key?") from ex

        assert isinstance(values, Tensor), f"scores must be a torch.Tensor, got {type(values)}"

        current_stage = trainer.state.stage
        logkey_prefix = f"{current_stage}/" if current_stage is not None else ""
        logkey = f"{logkey_prefix}histogram-of-{self.key}"

        # requirement from wandb.Histogram()
        values = values.detach().numpy() if values.requires_grad else values

        if self.mode == LOG_HISTOGRAM_MODE_LOG:
            wandb.log({logkey: wandb.Histogram(values)})

        elif self.mode == LOG_HISTOGRAM_MODE_SUMMARY:
            wandb.run.summary.update({logkey: wandb.Histogram(values)})

        else:
            raise ValueError(f"log_mode must be one of {LOG_HISTOGRAM_MODES}, got '{self.mode}'")


class LogHistogramsSuperposedCallback(
    MultiStageCallbackMixin,
    LastEpochOutputsDependentCallbackMixin,
    RandomCallbackMixin,
    pl.Callback,
):

    @staticmethod
    def cli_add_arguments(parser: ArgumentParser, histogram_of: str, stage: str, **defaults) -> CliArgNameMap:
        
        assert stage in MultiStageCallbackMixin.ACCEPTED_STAGES, f"stage must be one of {MultiStageCallbackMixin.ACCEPTED_STAGES}, got {stage}"
        assert isinstance(stage, str), f"stage must be a str, got {type(stage)}"
        assert isinstance(histogram_of, str), f"histogram_of must be a str, got {type(histogram_of)}"
        assert re.match(REGEXSTR_ASSERT_VARIABLE_NAME, histogram_of), f"histogram_of must be a valid variable name, got {histogram_of}"
        
        parser.description = f"Log histograms superposed per class of {histogram_of} for stage={stage}."
        
        cli_arg_name_map = dict()
                
        def cliname(argname):
            cliname = f"histograms_supperposed_{histogram_of}_{stage}_{argname}"
            cli_arg_name_map[cliname] = argname
            return cliname
        
        parser.add_argument(
            f'--{cliname("values_key")}', 
            type=str, 
        )
        
        parser.add_argument(
            f'--{cliname("gt_key")}', 
            type=str, 
        )
        
        parser.add_argument(
            f'--{cliname("mode")}', 
            type=str, choices=LOG_HISTOGRAM_MODES,
        )
        
        parser.add_argument(
            f'--{cliname("limit_points")}', 
            type=int, 
        )
        
        # these shouldnt be changed
        parser.add_argument(f'--{cliname("stage")}', type=str, default=stage,)
        # the seed here can be fixed, no need to follow the seed of the other
        # random stuff because in anycase the training is already random, and 
        # this randomness is just for the sampling of points used to estimate the ROC
        # it's not a big deal if all trainings use the same seed
        parser.add_argument(f'--{cliname("python_generator")}', type=create_python_random_generator, default=create_python_random_generator(0),)
        parser.set_defaults(**{cliname(argname): argval for argname, argval in defaults.items()})
        
        return cli_arg_name_map
    
    def __init__(self, values_key: str, gt_key: str, mode: str, limit_points: int, stage: RunningStageOrStr, python_generator: random.Random,):
        """
        Args:
            values_key & gt_key: inside the last_epoch_outputs, how are the values and their respective labels called?
        """
        super().__init__()

        assert values_key != "", f"scores_key must not be empty"
        assert gt_key != "", f"gt_key must not be empty"
        assert values_key != gt_key, f"scores_key and gt_key must be different, got {values_key} and {gt_key}"

        assert mode in LOG_HISTOGRAM_MODES, f"log_mode must be one of {LOG_HISTOGRAM_MODES}, got '{mode}'"
        assert mode is not None, f"log_mode must not be None, just dont add the callback to the trainer :)"

        if limit_points is not None:
            assert limit_points > 0, f"limit_points must be > 0 or None, got {limit_points}"

        self.values_key = values_key
        self.gt_key = gt_key
        self.mode = mode
        self.limit_points = limit_points
        
        # mixin args
        self.stage = stage
        self.python_generator = python_generator

    def setup(self, trainer, pl_module, stage=None):
        self.setup_validate_plmodule_last_epoch_outputs(trainer, pl_module, stage=stage)

    def _multi_stage_epoch_end_do(self, trainer, pl_module):
        self._log_histograms_supperposed_per_class(trainer, pl_module)

    def _log_histograms_supperposed_per_class(self, trainer: pl.Trainer, pl_module: pl.LightningModule):

        try:
            values = pl_module.last_epoch_outputs[self.values_key].detach()
            gt = pl_module.last_epoch_outputs[self.gt_key]

        except KeyError as ex:
            msg = ex.args[0]
            if self.values_key not in msg and self.gt_key not in msg:
                raise ex
            raise ValueError(f"pl_module.last_epoch_outputs should have the keys self.values_key={self.values_key} and self.gt_key={self.gt_key}, did you configure the model correctly? or passed me the wrong keys?") from ex

        assert isinstance(values, Tensor), f"values must be a torch.Tensor, got {type(values)}"
        assert isinstance(gt, Tensor), f"gt must be a torch.Tensor, got {type(gt)}"

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

        table = wandb.Table(data=table, columns=[self.values_key, "gt"])

        if self.mode == LOG_HISTOGRAM_MODE_LOG:
            wandb.log({logkey: table})

        elif self.mode == LOG_HISTOGRAM_MODE_SUMMARY:
            raise NotImplementedError(f"mode {self.mode} was having display problems in wandb")
            wandb.run.summary.update({logkey: table})

        else:
            raise ValueError(f"mode must be one of {LOG_HISTOGRAM_MODES}, got '{self.mode}'")


class DataloaderPreviewCallback(pl.Callback):

    def __init__(self, dataloader, n_samples=5,  stage="train"):
        """the keys loggeda are `{logkey_prefix}/anomalous` and `{logkey_prefix}/normal`"""
        super().__init__()
        assert isinstance(dataloader, torch.utils.data.DataLoader), f"dataloader must be a torch.utils.data.DataLoader, got {type(dataloader)}"
        assert n_samples > 0, f"n_samples must be > 0, got {n_samples}"
        self.dataloader = dataloader
        self.n_samples = n_samples
        
        assert isinstance(stage, RunningStage) or isinstance(stage, str), f"stage must be a RunningStage or a string, got {type(stage)}"
        ACCEPTED_STAGES = (
            RunningStage.TRAINING,
            RunningStage.VALIDATING,
            RunningStage.TESTING,
        )
        assert stage in ACCEPTED_STAGES, f"stage must be one of {ACCEPTED_STAGES}, got {stage}"
        self.stage = stage

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
        ) = data_dev01_bis.generate_dataloader_images(self.dataloader, nimages_perclass=self.n_samples)

        wandb.log({
            f"{self.stage}/preview_normal": [
                wandb.Image(img, caption=[f"normal {idx:03d}"], masks=self._get_mask_dict(mask))
                for idx, (img, mask) in enumerate(zip(norm_imgs, norm_gtmaps))
            ],
            f"{self.stage}/preview_anomalous": [
                wandb.Image(img, caption=[f"anomalous {idx:03d}"], masks=self._get_mask_dict(mask))
                for idx, (img, mask) in enumerate(zip(anom_imgs, anom_gtmaps))
            ],
        })


class LogImageHeatmapTableCallback(
    MultiStageCallbackMixin,
    LastEpochOutputsDependentCallbackMixin,
    RandomCallbackMixin,
    pl.Callback,
):

    @staticmethod
    def cli_add_arguments(parser: ArgumentParser, stage: str, **defaults) -> CliArgNameMap:
        
        assert stage in MultiStageCallbackMixin.ACCEPTED_STAGES, f"stage must be one of {MultiStageCallbackMixin.ACCEPTED_STAGES}, got {stage}"
        assert isinstance(stage, str), f"stage must be a str, got {type(stage)}"
        
        parser.description = f"Log image/heatmap table for stage={stage}."
        
        cli_arg_name_map = dict()
                
        def cliname(argname):
            cliname = f"imageheatmap_{stage}_{argname}"
            cli_arg_name_map[cliname] = argname
            return cliname
        
        parser.add_argument(
            f'--{cliname("imgs_key")}', 
            type=str, 
        )
        
        parser.add_argument(
            f'--{cliname("scores_key")}', 
            type=str, 
        )
        
        parser.add_argument(
            f'--{cliname("masks_key")}', 
            type=str, 
        )
        
        parser.add_argument(
            f'--{cliname("labels_key")}', 
            type=str, 
        )
        
        parser.add_argument(
            f'--{cliname("nsamples")}', 
            type=int, 
            help="How many images per class to log",
        )
        
        parser.add_argument(
            f'--{cliname("resolution")}', 
            type=none_or_int, 
            help="If None, the original resolution is used. Otherwise, the resolution is scaled to this value.",
        )
        
        parser.add_argument(
            f'--{cliname("heatmap_normalization")}', 
            type=str, choices=HEATMAP_NORMALIZATION_CHOICES,
        )
        
        parser.add_argument(
            f'--{cliname("min_max_percentiles")}', 
            type=float, nargs=2, 
            help="Percentile values for the contrast of the heatmap: min/max", 
        )
                
        # these shouldnt be changed
        parser.add_argument(f'--{cliname("stage")}', type=str, default=stage,)
        # the seed here can be fixed, no need to follow the seed of the other
        # random stuff because in anycase the training is already random, and 
        # this randomness is just for the sampling of points used to estimate the ROC
        # it's not a big deal if all trainings use the same seed
        parser.add_argument(f'--{cliname("python_generator")}', type=create_python_random_generator, default=create_python_random_generator(0),)
        parser.set_defaults(**{cliname(argname): argval for argname, argval in defaults.items()})
        
        return cli_arg_name_map
    
    def __init__(
        self,
        imgs_key: str,
        scores_key: str,
        masks_key: str,
        labels_key: str,
        nsamples: int,
        resolution: Optional[int],
        heatmap_normalization: str,
        min_max_percentiles: Optional[Tuple[float, float]],
        # mixin args
        stage: RunningStageOrStr,
        python_generator: random.Random,
    ):
        """
        it is assumed that the number and order of images is the same every epoch
        """
        super().__init__()

        assert imgs_key != "", f"imgs_key must be provided"
        assert scores_key != "", f"heatmaps_key must not be empty"
        assert masks_key != "", f"masks_key must not be empty"
        assert labels_key != "", f"labels_key must not be empty"

        # make sure there is no repeated key
        keys = [imgs_key, scores_key, masks_key, labels_key]
        assert len(keys) == len(set(keys)), f"keys must be unique, got {keys}"

        assert nsamples >= 0, f"nsamples must be >= 0, got {nsamples}"
        if nsamples == 0:
            warnings.warn(f"nsamples={nsamples} is 0, no images will be logged for stage={stage}")

        assert heatmap_normalization in HEATMAP_NORMALIZATION_CHOICES, f"heatmap_normalization must be one of {HEATMAP_NORMALIZATION_CHOICES}, got {heatmap_normalization}"

        if heatmap_normalization == HEATMAP_NORMALIZATION_PERCENTILES_IN_EPOCH:
            assert min_max_percentiles is not None, f"min_max_percentiles must be provided when heatmap_normalization is {HEATMAP_NORMALIZATION_PERCENTILES_IN_EPOCH}"
            assert len(min_max_percentiles) == 2, f"min_max_percentiles must be a tuple of length 2, got {min_max_percentiles}"
            min_max_percentiles = tuple(min_max_percentiles)
            min_percentile, max_percentile = min_max_percentiles
            assert 0 <= min_percentile < max_percentile <= 100, f"min_max_percentiles must be between 0 and 100 and min must be < max, got {min_max_percentiles}"

        if resolution is not None:
            assert resolution > 0, f"resolution must be > 0, got {resolution}"

        self.imgs_key = imgs_key
        self.scores_key = scores_key
        self.masks_key = masks_key
        self.labels_key = labels_key
        self.nsamples = nsamples
        self.heatmap_normalization = heatmap_normalization
        self.resolution = resolution
        self.min_max_percentiles = min_max_percentiles

        # lazy initialization because we don't know the batch size in advance
        self.selected_instances_indices = None
        
        # mixin args
        self.stage = stage
        self.python_generator = python_generator

    def setup(self, trainer, pl_module, stage=None):
        self.setup_validate_plmodule_last_epoch_outputs(trainer, pl_module, stage=stage)

    def _multi_stage_epoch_end_do(self, trainer, pl_module):

        try:
            imgs: Tensor = pl_module.last_epoch_outputs[self.imgs_key].detach()
            scores: Tensor = pl_module.last_epoch_outputs[self.scores_key].detach()
            masks: Tensor = pl_module.last_epoch_outputs[self.masks_key]
            labels: Tensor = pl_module.last_epoch_outputs[self.labels_key]

        except KeyError as ex:
            msg = ex.args[0]
            keys = [self.imgs_key, self.scores_key, self.masks_key, self.labels_key]

            # another exception not from here?
            if all(key not in msg for key in keys):
                raise ex

            raise ValueError(f"pl_module.last_epoch_outputs should have the keys={keys}, did you configure the model correctly? or passed me the wrong keys?") from ex

        if self.selected_instances_indices is None:

            labels = labels.numpy()
            normals_indices = np.where(labels == NOMINAL_TARGET)[0]
            anomalies_indices = np.where(labels == ANOMALY_TARGET)[0]

            if len(normals_indices) > self.nsamples:
                # python_generator.sample() is without replacement
                normals_indices: List[int] = self.python_generator.sample(normals_indices.tolist(), self.nsamples)

            if len(anomalies_indices) > self.nsamples:
                # python_generator.sample() is without replacement
                anomalies_indices: List[int] = self.python_generator.sample(anomalies_indices.tolist(), self.nsamples)

            self.selected_instances_indices = [*normals_indices, *anomalies_indices]

        self._log(trainer, pl_module, imgs, scores, masks, labels, self.selected_instances_indices)

    def _log(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        imgs: Tensor,
        scores: Tensor,
        masks: Tensor,
        labels: Tensor,
        selected_instances_indices: List[int],
    ):

        assert isinstance(imgs, Tensor), f"imgs must be a torch.Tensor, got {type(imgs)}"
        assert isinstance(scores, Tensor), f"scores must be a torch.Tensor, got {type(scores)}"
        assert isinstance(masks, Tensor), f"binary_gt must be a torch.Tensor, got {type(masks)}"

        assert scores.shape == masks.shape, f"scores and binary_gt must have the same shape, got {scores.shape} and {masks.shape}"
        assert imgs.shape[0] == scores.shape[0], f"imgs and scores must have the same number of samples, got {imgs.shape[0]} and {scores.shape[0]}"
        assert imgs.shape[2:] == scores.shape[2:], f"imgs and scores must have the same shape, got {imgs.shape} and {scores.shape}"

        assert (imgs >= 0).all(), f"imgs must be > 0"
        assert (imgs <= 1).all(), f"imgs must be <= 1"

        unique_gt_values = tuple(sorted(torch.unique(masks)))
        assert unique_gt_values in ((0,), (1,), (0, 1)), f"binary_gt must have only 0 and 1 values, got {unique_gt_values}"

        assert (scores >= 0).all(), f"scores must be >= 0, got {scores}"

        # i need this variable to the decide if an exception is raised or not later in the other if-else
        normalized = True
        
        # when something goes wrong, and another normalization is tried
        # this variable is set so we can use the right value in the wandb table
        fallback_heatmap_normalization = None
        
        @torch.no_grad()
        def normalize_minmax(scores_):
            min_ = scores_.min()
            return (scores_ - min_) / (scores_.max() - min_)
        
        @torch.no_grad()
        def normalize_inepoch_percentiles(scores_, minmax_percentiles):
            min_, max_ = torch.quantile(scores_.view(-1), torch.tensor(minmax_percentiles) / 100)  # "/100" percentile -> quantile 
            return (scores_ - min_) / (max_ - min_)
        
        @torch.no_grad()
        def normalize_percentiles_adaptive(scores_, masks_):
            
            try:
                clipmin, clipmax = find_scores_clip_values_from_empircal_cdf(
                    scores_normal=scores_[masks_ == 0], 
                    scores_anomalous=scores_[masks_ == 1]
                )
                return (scores_ - clipmin) / (clipmax - clipmin)

            except AdaptiveClipError as ex:
                
                FALLBACK_PERCENTILES = (3, 97)
                
                warnings.warn(f"AdaptiveClipError: clipping could not be applied, using normalization '{HEATMAP_NORMALIZATION_PERCENTILES_IN_EPOCH}' instead with in-epoch fallback percentiles {FALLBACK_PERCENTILES}, error: {ex}", stacklevel=2)
                
                nonlocal fallback_heatmap_normalization
                fallback_heatmap_normalization = HEATMAP_NORMALIZATION_PERCENTILES_IN_EPOCH
                
                return normalize_inepoch_percentiles(scores_, FALLBACK_PERCENTILES)
        
        @torch.no_grad()
        def normalize_minmax_instance(scores_):
            min_ = scores_.min(dim=(1, 2, 3), keepdim=True)
            max_ = scores_.max(dim=(1, 2, 3), keepdim=True)
            return (scores_ - min_) / (max_ - min_)
        
        if self.heatmap_normalization == HEATMAP_NORMALIZATION_MINMAX_IN_EPOCH:
            scores = normalize_minmax(scores)

        elif self.heatmap_normalization == HEATMAP_NORMALIZATION_PERCENTILES_IN_EPOCH:
            scores = normalize_inepoch_percentiles(scores, self.min_max_percentiles)
        
        elif self.heatmap_normalization == HEATMAP_NORMALIZATION_PERCENTILES_ADAPTIVE_CDF_BASED_IN_EPOCH:
            scores = normalize_percentiles_adaptive(scores, masks)
                
        else:
            normalized = False
            # dont raise an error here because there is still the case below,
            # where the normalizations are per instance, so i avoid computation on useless data
        
        # imgs: [instances, channels=3, height, width]
        # scores: [instances, 1, height, width]
        # masks: [instances, 1, height, width]
        # labels: [instances]
        imgs, scores, masks, labels = imgs[selected_instances_indices], scores[selected_instances_indices], masks[selected_instances_indices], labels[selected_instances_indices]
        
        if self.heatmap_normalization == HEATMAP_NORMALIZATION_MINMAX_INSTANCE:
            scores = normalize_minmax_instance(scores)
                
        elif not normalized:
            raise NotImplementedError(f"heatmap_normalization={self.heatmap_normalization} is not implemented")

        # self.resolution == None means: "dont change the resolution"
        if self.resolution is not None:
            img_h, img_w = imgs.shape[-2:]
            if img_h != img_w:
                raise NotImplementedError(f"imgs must be square, got {img_h}x{img_w}")
            if img_w != self.resolution:
                imgs = TFT.resize(imgs, self.resolution, interpolation="bilinear",)
                scores = TFT.resize(scores, self.resolution,interpolation="bilinear",)
                masks = TFT.resize(masks, self.resolution,interpolation="nearest",)

        imgs = imgs.detach().cpu().numpy().transpose(0, 2, 3, 1)  # [instances, height, width, channels]
        scores = scores.detach().cpu().numpy().transpose(0, 2, 3, 1)  # [instances, height, width, channels]
        masks = masks.detach().cpu().squeeze(1).numpy()

        class_labels_dict = {NOMINAL_TARGET: "normal", ANOMALY_TARGET: "anomalous",}

        table = wandb.Table(columns=["idx", "idx-in-epoch", "label", "image", "heatmap", "normalization"])

        for idx, idx_in_epoch in enumerate(selected_instances_indices):

            # their repectives in the singular lose the 1st dimension (instances)
            img, score_map, mask, label = imgs[idx], scores[idx], masks[idx], labels[idx]

            wandb_img = wandb.Image(
                img,
                masks=dict(ground_truth=dict(mask_data=mask, class_labels=class_labels_dict,))
            )
            wandb_heatmap = wandb.Image(
                score_map,
                masks=dict(ground_truth=dict(mask_data=mask, class_labels=class_labels_dict,))
            )
            normalization_used = fallback_heatmap_normalization if fallback_heatmap_normalization is not None else self.heatmap_normalization
            table.add_data(idx, idx_in_epoch, label, wandb_img, wandb_heatmap, normalization_used)

        current_stage = trainer.state.stage
        logkey_prefix = f"{current_stage}/" if current_stage is not None else ""
        table_logkey = f"{logkey_prefix}images-heatmaps-table"
        wandb.log({table_logkey: table})


class LogPercentilesPerClassCallback(
    MultiStageCallbackMixin,
    LastEpochOutputsDependentCallbackMixin,
    pl.Callback,
):
  
    @staticmethod
    def cli_add_arguments(parser: ArgumentParser, percentiles_of: str, stage: str, **defaults) -> CliArgNameMap:
        
        assert stage in MultiStageCallbackMixin.ACCEPTED_STAGES, f"stage must be one of {MultiStageCallbackMixin.ACCEPTED_STAGES}, got {stage}"
        assert isinstance(stage, str), f"stage must be a str, got {type(stage)}"
        assert isinstance(percentiles_of, str), f"percentiles_of must be a str, got {type(percentiles_of)}"
        assert re.match(REGEXSTR_ASSERT_VARIABLE_NAME, percentiles_of), f"percentiles_of must be a valid variable name, got {percentiles_of}"
        
        parser.description = f"Log percentiles per class of {percentiles_of} for stage={stage}."
        
        cli_arg_name_map = dict()
                
        def cliname(argname):
            cliname = f"percentiles_{percentiles_of}_{stage}_{argname}"
            cli_arg_name_map[cliname] = argname
            return cliname
        
        parser.add_argument(
            f'--{cliname("values_key")}', 
            type=str, 
        )
        
        parser.add_argument(
            f'--{cliname("gt_key")}', 
            type=str, 
        )
        
        parser.add_argument(
            f'--{cliname("percentiles")}', 
            type=bool, 
        )
                
        # these shouldnt be changed
        parser.add_argument(f'--{cliname("stage")}', type=str, default=stage,)
        # the seed here can be fixed, no need to follow the seed of the other
        # random stuff because in anycase the training is already random, and 
        # this randomness is just for the sampling of points used to estimate the ROC
        # it's not a big deal if all trainings use the same seed
        parser.add_argument(f'--{cliname("python_generator")}', type=create_python_random_generator, default=create_python_random_generator(0),)
        parser.set_defaults(**{cliname(argname): argval for argname, argval in defaults.items()})
        
        return cli_arg_name_map

    def __init__(self, values_key: str, gt_key: str, percentiles: Tuple[float, ...], stage: RunningStageOrStr,):
        """
        Args:
            values_key & gt_key: inside the last_epoch_outputs, how are the values and their respective labels called?
        """
        super().__init__()
        assert values_key != "", f"values_key must not be empty"
        assert gt_key != "", f"gt_key must not be empty"
        assert values_key != gt_key, f"values_key and gt_key must be different, got {values_key} and {gt_key}"
        assert len(percentiles) > 0, f"got {len(percentiles)}"
        assert all(0 <= p <= 100 for p in percentiles), f"got {percentiles}"
        self.values_key = values_key
        self.gt_key = gt_key
        self.percentiles = percentiles
        
        # mixin args
        self.stage = stage

    def setup(self, trainer, pl_module, stage=None):
        self.setup_validate_plmodule_last_epoch_outputs(trainer, pl_module, stage=stage)

    def _multi_stage_epoch_end_do(self, trainer, pl_module):

        try:
            values = pl_module.last_epoch_outputs[self.values_key].detach()
            gt = pl_module.last_epoch_outputs[self.gt_key]

        except KeyError as ex:
            msg = ex.args[0]
            if self.values_key not in msg and self.gt_key not in msg:
                raise ex
            raise ValueError(f"pl_module.last_epoch_outputs should have the keys self.values_key={self.values_key} and self.gt_key={self.gt_key}, did you configure the model correctly? or passed me the wrong keys?") from ex

        self._log(trainer, pl_module, values, gt)

    def _log(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, values: Tensor, gt: Tensor
    ):

        assert isinstance(values, Tensor), f"values must be a torch.Tensor, got {type(values)}"
        assert isinstance(gt, Tensor), f"gt must be a torch.Tensor, got {type(gt)}"

        assert values.shape == gt.shape, f"values and gtg must have the same shape, got {values.shape} and {gt.shape}"

        values_normal = values[gt == NOMINAL_TARGET]
        values_anomalous = values[gt == ANOMALY_TARGET]

        current_stage = trainer.state.stage
        logkey_prefix = f"{current_stage}/" if current_stage is not None else ""
        logkey = f"{logkey_prefix}percentiles-per-class-of-{self.values_key}"

        table = wandb.Table(columns=["percentile", "normal", "anomalous"])

        for perc in self.percentiles:
            table.add_data(perc, np.percentile(values_normal, q=perc), np.percentile(values_anomalous, q=perc))

        wandb.log({logkey: table})


class LogPerInstanceMeanCallback(
    MultiStageCallbackMixin,
    LastEpochOutputsDependentCallbackMixin,
    pl.Callback,
):

    @staticmethod
    def cli_add_arguments(parser: ArgumentParser, mean_of: str, stage: str, **defaults) -> CliArgNameMap:
        
        assert stage in MultiStageCallbackMixin.ACCEPTED_STAGES, f"stage must be one of {MultiStageCallbackMixin.ACCEPTED_STAGES}, got {stage}"
        assert isinstance(stage, str), f"stage must be a str, got {type(stage)}"
        assert isinstance(mean_of, str), f"percentiles_of must be a str, got {type(mean_of)}"
        assert re.match(REGEXSTR_ASSERT_VARIABLE_NAME, mean_of), f"percentiles_of must be a valid variable name, got {mean_of}"
        
        parser.description = f"Log per instance mean of {mean_of} for stage={stage}."
        
        cli_arg_name_map = dict()
                
        def cliname(argname):
            cliname = f"perinstance_mean_{mean_of}_{stage}_{argname}"
            cli_arg_name_map[cliname] = argname
            return cliname
        
        parser.add_argument(
            f'--{cliname("values_key")}', 
            type=str, 
        )
        
        parser.add_argument(
            f'--{cliname("labels_key")}', 
            type=str, 
        )
                
        # these shouldnt be changed
        parser.add_argument(f'--{cliname("stage")}', type=str, default=stage,)
        parser.set_defaults(**{cliname(argname): argval for argname, argval in defaults.items()})
        
        return cli_arg_name_map

    def __init__(self, values_key: str, labels_key: str, stage: RunningStageOrStr,):
        """
        Args:
            values_key & gt_key: inside the last_epoch_outputs, how are the values and their respective labels called?
        """
        super().__init__()
        assert values_key != "", f"values_key must not be empty"
        assert labels_key != "", f"labels_key must not be empty"
        assert values_key != labels_key, f"values_key and labels_key must be different, got {values_key} and {labels_key}"
        
        self.values_key = values_key
        self.labels_key = labels_key
        
        # mixin args
        self.stage = stage

    def setup(self, trainer, pl_module, stage=None):
        self.setup_validate_plmodule_last_epoch_outputs(trainer, pl_module, stage=stage)

    def _multi_stage_epoch_end_do(self, trainer, pl_module):

        try:
            values = pl_module.last_epoch_outputs[self.values_key].detach()
            labels = pl_module.last_epoch_outputs[self.labels_key]

        except KeyError as ex:
            msg = ex.args[0]
            if self.values_key not in msg and self.labels_key not in msg:
                raise ex
            raise ValueError(f"pl_module.last_epoch_outputs should have the keys self.values_key={self.values_key} and self.labels_key={self.labels_key}, did you configure the model correctly? or passed me the wrong keys?") from ex

        self._log(trainer, pl_module, values, labels)

    def _log(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, values: Tensor, labels: Tensor
    ):
        assert isinstance(values, Tensor), f"values must be a torch.Tensor, got {type(values)}"
        assert isinstance(labels, Tensor), f"labels must be a torch.Tensor, got {type(labels)}"

        assert values.shape[0] == labels.shape[0], f"values and labels must have the same number of instances, got {values.shape[0]} and {labels.shape[0]}"

        ninstances = values.shape[0]
        table = torch.cat(
            [
                torch.arange(ninstances).unsqueeze(1),
                values.mean(dim=(1, 2, 3)).unsqueeze(1),
                labels.unsqueeze(1),
            ],
            dim=1,
        )

        table = wandb.Table(
            data=table.numpy(),
            columns=["idx", f"image-mean", "image-label"]
        )

        current_stage = trainer.state.stage
        logkey_prefix = f"{current_stage}/" if current_stage is not None else ""
        logkey = f"{logkey_prefix}perinstance-mean-{self.values_key}"

        wandb.log({logkey: table})


class LearningRateLoggerCallback(pl.Callback):
    
    def __init__(self):
        super().__init__()

    def on_train_epoch_start(self, trainer, pl_module):
        for idx, scheduler in enumerate(trainer.lr_schedulers):
            # joao: idk why this [0] is necessary
            current_lr = scheduler['scheduler'].get_last_lr()[0]
            trainer.model.log(f"train/learning_rate_scheduler_idx={idx}", current_lr)


class LogHistAvgAtEpochEnd(
    MultiStageCallbackMixin,
    pl.Callback,
):
    """
    At the end of an epoch, fetch the history of the given metrics and log their average over the history.
    BUG: it is always missing the current epoch and will log the average of the history up to the last epoch.
    """
    
    @staticmethod
    def cli_add_arguments(parser: ArgumentParserOrArgumentGroup, stage: str, **defaults) -> CliArgNameMap:
        
        assert stage in MultiStageCallbackMixin.ACCEPTED_STAGES, f"stage must be one of {MultiStageCallbackMixin.ACCEPTED_STAGES}, got {stage}"
        assert isinstance(stage, str), f"stage must be a str, got {type(stage)}"
        
        parser.description = f"At the end of an epoch, fetch the history of the given metrics and log their average over the history for stage={stage}."
        
        cli_arg_name_map = dict()
                
        def cliname(argname):
            cliname = f"log_histavg_{stage}_{argname}"
            cli_arg_name_map[cliname] = argname
            return cliname
        
        parser.add_argument(
            f'--{cliname("metric_names")}', 
            type=str, action='extend', nargs='+',
        )
           
        # these shouldnt be changed
        parser.add_argument(f'--{cliname("stage")}', type=str, default=stage,)
        parser.set_defaults(**{cliname(argname): argval for argname, argval in defaults.items()})
        
        return cli_arg_name_map
        
    def __init__(
        self,
        metric_names: Tuple[str, ...],
        # mixin args
        stage: RunningStageOrStr,
    ):
        super().__init__()
        
        if len(metric_names) == 0:
            warnings.warn(f"no metrics were passed to LogHistAvgAtEndOfEpoch, this is probably not what you want, nothing will be logged from here!", stacklevel=2)

        for mn in metric_names:
            assert isinstance(mn, str), f"metric_names must be a tuple of str, got {type(mn)}"
            assert mn != "", f"metric_names must not be empty"
            assert mn.startswith(f"{stage}/"), f"metric_names must start with {stage}/, got {mn}"
        
        self.metric_names = metric_names    
        self._apirun = None

        # from mixins (validations already done in the mixin)
        self.stage = stage
        
    def _multi_stage_epoch_end_do(self, trainer, pl_module):
        self._log_histavg_metrics(trainer)
        
    def _log_histavg_metrics(self, trainer):
        
        nloggers = len(trainer.model.loggers)
        
        if nloggers == 0:
            warnings.warn(f"no loggers found in LogHistAvgAtEndOfEpoch (in trainer.model.loggers), nothing will be logged from here!", stacklevel=1)
            return  
        
        elif nloggers > 1:
            
            wandb_loggers = [l for l in trainer.model.loggers if isinstance(l, WandbLogger)]
            
            if len(wandb_loggers) > 1:
                raise ValueError(f"there must be only one WandbLogger in the model, got {len(wandb_loggers)}")
        
            wandb_logger = wandb_loggers[0] 
        
        else:
            wandb_logger = trainer.model.loggers[0]

        def get_run_fullid(run_) -> str:
            return f"{run_.entity}/{run_.project}/{run_.id}"
        
        records = wandb.Api().run(get_run_fullid(wandb_logger.experiment)).scan_history(keys=self.metric_names)
        
        if records is None or len(records := list(records)) == 0:
            warnings.warn(f"no records found in wandb.run.scan_history", stacklevel=1)
            return 
        
        histavg_metrics = pd.DataFrame.from_records(data=records).mean(axis=0).to_dict()    
        histavg_metrics = {f"{k}-histavg": v for k, v in histavg_metrics.items()}
        
        for k, v in histavg_metrics.items():
            trainer.model.log(k, v)
