#!/usr/bin/env python
# coding: utf-8

import functools
import random
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.trainer.states import RunningStage
from sklearn.metrics import (roc_auc_score, roc_curve)
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
    from wandb.plots.utils import test_missing, test_types

    import wandb
    from wandb import util

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


class LogRocCallback(
    LastEpochOutputsDependentCallbackMixin, 
    RandomCallbackMixin, 
    pl.Callback,
):
    
    ACCEPTED_STAGES = [
        RunningStage.TRAINING,
        RunningStage.VALIDATING,
        RunningStage.TESTING,
    ]
    
    @RandomCallbackMixin.init_python_generator
    def __init__(
        self, 
        stage: Union[str, RunningStage],
        scores_key: str,
        gt_key: str,
        log_curve: bool = False, 
        limit_points: int = 3000,
    ):
        """
        Log the ROC curve (and it AUC) on val/test batches.
        
        Args:
            limit_points (int, optional): sample this number of points from all the available scores, if None, then no limitation is applied. Defaults to 3000.
        """
        super().__init__()
        
        assert stage in self.ACCEPTED_STAGES, f"stage must be one of {self.ACCEPTED_STAGES}, got {stage}"
        
        assert scores_key != "", f"scores_key must not be empty"
        assert gt_key != "", f"gt_key must not be empty"
        assert scores_key != gt_key, f"scores_key and gt_key must be different, got {scores_key} and {gt_key}"
        
        if limit_points is not None:
            assert limit_points > 0, f"limit_points must be > 0 or None, got {limit_points}"
            
        self.stage = stage
        self.scores_key = scores_key
        self.gt_key = gt_key
        self.log_curve = log_curve
        self.limit_points = limit_points
    
    @LastEpochOutputsDependentCallbackMixin.setup_verify_plmodule_with_last_epoch_outputs
    def setup(self, trainer, pl_module, stage=None):
        pass  # just let the mixin do its job
    
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

            # i copied and adapted wandb.plot.roc_curve()
            wandb.log({curve_logkey: roc_curve(binary_gt, scores, labels=["anomalous"])})
           
        trainer.model.log(auc_logkey, roc_auc_score(binary_gt, scores))
    
    @staticmethod
    def should_log(hood_stage, self_stage, trainer_stage, ):
        return self_stage == hood_stage and trainer_stage == hood_stage
    
    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        if self.should_log(RunningStage.TRAINING, self.stage, trainer.state.stage):
            self._log_roc(trainer, pl_module)
    
    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        if self.should_log(RunningStage.VALIDATING, self.stage, trainer.state.stage): 
            self._log_roc(trainer, pl_module)

    def on_test_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        if self.should_log(RunningStage.TESTING, self.stage, trainer.state.stage): 
            self._log_roc(trainer, pl_module)


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
