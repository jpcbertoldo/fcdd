#!/usr/bin/env python
# coding: utf-8

# In[]:
# # train mvtec
# 
# i want to have a simpler script to integrate to wandb and later adapt it to unetdd

# In[]:
# # imports

# In[1]:


import json
import os.path as pt
import random
import time
from argparse import ArgumentParser
from collections import namedtuple
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Union
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from fcdd.models.bases import ReceptiveNet
from fcdd.util.logging import Logger
from pytorch_lightning.loggers import WandbLogger
from torch import Tensor, gt
from torch.hub import load_state_dict_from_url
import time

import wandb

import mvtec_dataset_dev01 as mvtec_datamodule
from mvtec_dataset_dev01 import MVTecAnomalyDetectionDataModule, create_seed
from train_mvtec_dev01_aux import single_save

from train_mvtec_dev01_aux import compute_gtmap_roc, compute_gtmap_pr

from torch.profiler import tensorboard_trace_handler

# # utils

def balance_labels(data: Tensor, labels: List[int], err=True) -> Tensor:
    """ balances data by removing samples for the more frequent label until both labels have equally many samples """
    lblset = list(set(labels))
    if err:
        assert len(lblset) == 2, 'binary labels required'
    else:
        assert len(lblset) <= 2, 'binary labels required'
        if len(lblset) == 1:
            return data
    l0 = (torch.from_numpy(np.asarray(labels)) == lblset[0]).nonzero().squeeze(-1).tolist()
    l1 = (torch.from_numpy(np.asarray(labels)) == lblset[1]).nonzero().squeeze(-1).tolist()
    if len(l0) > len(l1):
        rmv = random.sample(l0, len(l0) - len(l1))
    else:
        rmv = random.sample(l1, len(l1) - len(l0))
    ids = [i for i in range(len(labels)) if i not in rmv]
    data = data[ids]
    return data


# In[]:

# ======================================== dataset ========================================

DATASET_MVTEC = "mvtec"
DATASET_CHOICES = (DATASET_MVTEC,)


def dataset_class_labels(dataset_name: str) -> List[str]:
    return {
        DATASET_MVTEC: mvtec_datamodule.CLASSES_LABELS,
    }[dataset_name]


def dataset_nclasses(dataset_name: str) -> int:
    return {
        DATASET_MVTEC: mvtec_datamodule.NCLASSES,
    }[dataset_name]


def dataset_class_index(dataset_name: str, class_name: str) -> int:
    return dataset_class_labels(dataset_name).index(class_name)

# ======================================== preprocessing ========================================

def dataset_preprocessing_choices(dataset_name: str) -> List[str]:
    return {
        DATASET_MVTEC: mvtec_datamodule.PREPROCESSING_CHOICES,
    }[dataset_name]


PREPROCESSING_DEFAULT_DEFAULT = 'aug1'


PREPROCESSING_CHOICES = tuple(set.union(*[
    set(dataset_preprocessing_choices(dataset_name)) 
    for dataset_name in DATASET_CHOICES
]))

# ======================================== supervise mode ========================================

def dataset_supervise_mode_choices(dataset_name: str) -> List[str]:
    return {
        DATASET_MVTEC: mvtec_datamodule.SUPERVISE_MODES,
    }[dataset_name]


SUPERVISE_MODE_DEFAULT_DEFAULT = 'synthetic-anomaly-confetti'

SUPERVISE_MODE_CHOICES = tuple(set.union(*[
    set(dataset_supervise_mode_choices(dataset_name)) 
    for dataset_name in DATASET_CHOICES
]))

# ======================================== pytorch lightning ========================================

LIGHTNING_ACCELERATOR_CPU = "cpu"
LIGHTNING_ACCELERATOR_GPU = "gpu"
LIGHTNING_ACCELERATOR_CHOICES = (LIGHTNING_ACCELERATOR_CPU, LIGHTNING_ACCELERATOR_GPU)

# In[]:
# # model

# In[]:
# ## new (torch lightning)

# In[]:
from pytorch_lightning import LightningModule

class FCDD_CNN224_VGG(LightningModule):
    """
    # VGG_11BN based net with most of the VGG layers having weights 
    # pretrained on the ImageNet classification task.
    # these weights get frozen, i.e., the weights will not get updated during training
    """
    """ Baseclass for FCDD networks, i.e. network without fully connected layers that have a spatial output """
    
    MODEL_DIR = Path(__file__).parent.parent.parent / 'data' / 'models'  # todo make an arg?
    
    def __init__(
        self, 
        # model
        in_shape: Tuple[int, int, int], 
        gauss_std: float,
        bias: bool, 
        pixel_level_loss: bool,
        # optimizer
        optimizer_name: str,
        lr: float,
        weight_decay: float,
        # scheduler
        scheduler_name: str,
        lr_sched_param: list,
        # else
        normal_class_label: str, 
    ):
        assert bias, 'VGG net is only supported with bias atm!'
        
        super().__init__()
        # super(ReceptiveNet, self).__init__(in_shape, bias)
        # ReceptiveNet.__init__(self, in_shape, bias)
        # LightningModule.__init__(self)
        
        self._receptive_field_net = ReceptiveNet(in_shape, bias)
        self.biases = bias
        self.in_shape = in_shape
        self.gauss_std = gauss_std
        
        state_dict = load_state_dict_from_url(
            torchvision.models.vgg.model_urls['vgg11_bn'],
            model_dir=str(self.MODEL_DIR),
        )
        features_state_dict = {
            k[9:]: v  # cut "features_" ?
            for k, v in state_dict.items() 
            if k.startswith('features')
        }

        self.features = nn.Sequential(
            self._receptive_field_net._create_conv2d(3, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            self._receptive_field_net._create_maxpool2d(2, 2),
            self._receptive_field_net._create_conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            self._receptive_field_net._create_maxpool2d(2, 2),
            self._receptive_field_net._create_conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            self._receptive_field_net._create_conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            self._receptive_field_net._create_maxpool2d(2, 2),
            # Frozen version freezes up to here
            self._receptive_field_net._create_conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            self._receptive_field_net._create_conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # CUT
            nn.MaxPool2d(2, 2),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2)
        )
        self.features.load_state_dict(features_state_dict)
        self.features = self.features[:-8]        
        # free the layers in the middle
        for m in self.features[:15]:
            for p in m.parameters():
                p.requires_grad = False
        
        self.conv_final = self._receptive_field_net._create_conv2d(512, 1, 1)
        
        self.save_hyperparameters()
        pass
        
    def forward(self, x):
        x = self.features(x)
        x = self.conv_final(x)
        return x
    
    def loss(self, inputs: Tensor, gtmaps: Tensor) -> Tensor:
        """ computes the FCDD """
        
        anomaly_score_maps = self(inputs) 
        
        loss_maps = anomaly_score_maps ** 2
        loss_maps = (loss_maps + 1).sqrt() - 1
        
        gauss_std = self.hparams["gauss_std"]
        loss_maps = self._receptive_field_net.receptive_upsample(loss_maps, reception=True, std=gauss_std, cpu=False)
        
        norm_loss_maps = (loss_maps * (1 - gtmaps))
        
        anom_loss_maps = -(((1 - (-loss_maps).exp()) + 1e-31).log())
        anom_loss_maps = anom_loss_maps * gtmaps
        
        loss_maps = norm_loss_maps + anom_loss_maps
        
        return anomaly_score_maps, loss_maps, loss_maps.mean()
            
    def configure_optimizers(self):
        
        # =================================== optimizer =================================== 
        optimizer_name = self.hparams['optimizer_name']
        lr = self.hparams['lr']
        weight_decay = self.hparams['weight_decay']
        
        if optimizer_name == 'sgd':
            return optim.SGD(
                self.parameters(), 
                lr=lr, 
                weight_decay=weight_decay, 
                momentum=0.9, 
                nesterov=True
            )
        
        elif optimizer_name == 'adam':
            optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        
        else:
            raise NotImplementedError('Optimizer type {} not known.'.format(optimizer_name))
        
        # ================================ scheduler ================================
        
        scheduler_name = self.hparams['scheduler_name']
        lr_sched_param = self.hparams['lr_sched_param']
        
        if scheduler_name == 'lambda':
            
            assert len(lr_sched_param) == 1, 'lambda scheduler needs one parameter' 
            assert 0 < lr_sched_param[0] <= 1, 'lambda scheduler parameter [0] must be in (0, 1]'
            
            scheduler = optim.lr_scheduler.LambdaLR(
                optimizer, 
                lambda ep: lr_sched_param[0] ** ep
            )
        
        elif scheduler_name == 'milestones':
            
            assert len(lr_sched_param) >= 2, 'milestones scheduler needs at least two parameters' 
            assert 0 < lr_sched_param[0] <= 1, 'milestones scheduler parameter [0] (gamma) must be in (0, 1]' 
            assert all([p > 1 for p in lr_sched_param[1:]]), 'milestones scheduler parameters [1:] (milestones) must be > 1'
            
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer, 
                gamma=lr_sched_param[0], 
                milestones=[int(s) for s in lr_sched_param[1:]], 
            )
            
        else:
            raise NotImplementedError(f'LR scheduler type {scheduler_name} not known.')
        
        return [optimizer], [scheduler]     
        
    def training_step(self, batch, batch_idx):
        inputs, labels, gtmaps = batch
        anomaly_scores_maps, loss_maps, loss = self.loss(inputs=inputs, gtmaps=gtmaps)
        
        # separate the loss in normal/anomaly
        with torch.no_grad():
            loss_normal = loss_maps[gtmaps == 0].mean()
            loss_anomaly = loss_maps[gtmaps == 1].mean()
            
        self.log("train/loss/normal", loss_normal, on_step=True, on_epoch=True)
        self.log("train/loss/anomaly", loss_anomaly, on_step=True, on_epoch=True)
        self.log("train/loss", loss, on_step=True, on_epoch=True)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        
        inputs, labels, gtmaps = batch
        anomaly_scores_maps, loss_maps, loss = self.loss(inputs=inputs, gtmaps=gtmaps)
        
        loss_normal = loss_maps[gtmaps == 0].mean()
        loss_anomaly = loss_maps[gtmaps == 1].mean()
        
        self.log("test/loss/normal", loss_normal, on_step=False, on_epoch=True)
        self.log("test/loss/anomaly", loss_anomaly, on_step=False, on_epoch=True)
        self.log("test/loss", loss, on_step=False, on_epoch=True)
        
        return inputs, labels, gtmaps, anomaly_scores_maps, loss_maps, loss
    
    def test_epoch_end(self, test_step_outputs):  
        # transpose the list of lists
        inputs, labels, gtmaps, anomaly_scores_maps, loss_maps, loss = list(map(list, zip(*test_step_outputs)))
        inputs = torch.cat(inputs, dim=0)
        labels = torch.cat(labels, dim=0)
        gtmaps = torch.cat(gtmaps, dim=0)
        anomaly_scores_maps = torch.cat(anomaly_scores_maps, dim=0)
        loss_maps = torch.cat(loss_maps, dim=0)
        loss = torch.tensor(loss)
        
        # heatmap_generation()
        
        # maybe i have to use this again...
        #get_original_gtmaps_normal_class()
        
        gtmap_roc = compute_gtmap_roc(
            anomaly_scores=anomaly_scores_maps,
            original_gtmaps=gtmaps,
            net=self, 
        )
        single_save(self.logger.save_dir, 'test.gtmap_roc', gtmap_roc)
        
        gtmap_pr = compute_gtmap_pr(
            anomaly_scores=anomaly_scores_maps,
            original_gtmaps=gtmaps,
            net=self, 
        )
        single_save(self.logger.save_dir, 'test.gtmap_pr', gtmap_pr)
            
        normal_class_label = self.hparams['normal_class_label']
        self.logger.experiment.log({
            "test/rocauc": gtmap_roc["auc"],
            "test/ap": gtmap_pr["ap"],
            
            # ========================== ROC CURVE ==========================
            # copied from wandb.plot.roc_curve()
            # debug=wandb.plot.roc_curve(),
            "test/roc_curve": wandb.plot_table(
                vega_spec_name="wandb/area-under-curve/v0",
                data_table=wandb.Table(
                    columns=["class", "fpr", "tpr"], 
                    data=[
                        [normal_class_label, fpr_, tpr_] 
                        for fpr_, tpr_ in zip(
                            gtmap_roc["fpr"], 
                            gtmap_roc["tpr"],
                        )
                    ],
                ),
                fields={"x": "fpr", "y": "tpr", "class": "class"},
                string_fields={
                    "title": "ROC curve",
                    "x-axis-title": "False Positive Rate (FPR)",
                    "y-axis-title": "True Positive Rate (TPR)",
                },
            ),
            # ========================== PR CURVE ==========================
            # copied from wandb.plot.pr_curve()
            # debug=wandb.plot.pr_curve(),
            "test/pr_curve": wandb.plot_table(
                vega_spec_name="wandb/area-under-curve/v0",
                data_table=wandb.Table(
                    columns=["class", "recall", "precision"], 
                    data=[
                        [normal_class_label, rec_, prec_] 
                        for rec_, prec_ in zip(
                            gtmap_pr["recall"], 
                            gtmap_pr["precision"],
                        )
                    ],
                ),
                fields={"x": "recall", "y": "precision", "class": "class"},
                string_fields={
                    "title": "PR curve",
                    "x-axis-title": "Recall",
                    "y-axis-title": "Precision",
                },
            )
        })
    
    # DEPRECATE DEPRECATE DEPRECATE DEPRECATE DEPRECATE DEPRECATE DEPRECATE DEPRECATE DEPRECATE DEPRECATE DEPRECATE DEPRECATE
    
    def anomaly_score(self, loss: Tensor) -> Tensor:
        """ This assumes the loss is already the anomaly score. If this is not the case, reimplement the method! """
        assert not self.training
        return loss
    
    def reduce_ascore(self, ascore: Tensor) -> Tensor:
        """ Reduces the anomaly score to be a score per image (detection). """
        assert not self.training
        return ascore.reshape(ascore.size(0), -1).mean(1)
    
    def set_reception(self, *args, **kwargs):
        return self._receptive_field_net.set_reception(*args, **kwargs)
    
    def receptive_upsample(self, *args, **kwargs):
        return self._receptive_field_net.receptive_upsample(*args, **kwargs)

    def reset_parameters(self, *args, **kwargs):
        return self._receptive_field_net.reset_parameters(*args, **kwargs)
   
    @property
    def reception(self):
        return self._receptive_field_net.reception

    @property
    def initial_reception(self):
        return self._receptive_field_net.initial_reception
    
# In[]:

MODEL_CLASSES = {
    klass.__name__: klass
    for klass in [FCDD_CNN224_VGG]
}


# In[]:
# # args

# In[2]:


def default_parser_config(parser: ArgumentParser) -> ArgumentParser:
    """
    Defines all the arguments for running an FCDD experiment.
    :param parser: instance of an ArgumentParser.
    :return: the parser with added arguments
    """


    # training parameters
    parser.add_argument('-e', '--epochs', type=int, default=200)
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3)
    parser.add_argument('-wd', '--weight-decay', type=float, default=1e-6)
    parser.add_argument(
        '--optimizer', type=str, default='sgd', choices=['sgd', 'adam'],
        help='The type of optimizer. Defaults to "sgd". '
    )
    parser.add_argument(
        '--scheduler', type=str, default='lambda', choices=['lambda', 'milestones'],
        help='The type of learning rate scheduler. Either "lambda", which reduces the learning rate each epoch '
             'by a certain factor, or "milestones", which sets the learning rate to certain values at certain '
             'epochs. Defaults to "lambda"'
    )
    parser.add_argument(
        '--lr-sched-param', type=float, nargs='*', default=[0.985],
        help='Sequence of learning rate scheduler parameters. '
             'For the "lambda" scheduler, just one parameter is allowed, '
             'which sets the factor the learning rate is reduced per epoch. '
             'For the "milestones" scheduler, at least two parameters are needed, '
             'the first determining the factor by which the learning rate is reduced at each milestone, '
             'and the others being each a milestone. For instance, "0.1 100 200 300" reduces the learning rate '
             'by 0.1 at epoch 100, 200, and 300. '
    )
    parser.add_argument(
        '-n', '--net', type=str, default='FCDD_CNN224_VGG', choices=MODEL_CLASSES.keys(),
        help='Chooses a network architecture to train.'
    )
    parser.add_argument('--no-bias', dest='bias', action='store_false', help='Uses no bias in network layers.')

    # 
    # ===================================== anomaly settings =====================================
    # 
    parser.add_argument(
        '--supervise-mode', type=str, default=SUPERVISE_MODE_DEFAULT_DEFAULT, choices=SUPERVISE_MODE_CHOICES,
        help='This determines the kind of artificial anomalies. '
    )
    
    parser.add_argument(
        '--real-anomaly-limit', type=int, default=np.infty,
        help='Determines the number of real anomalous images used. '
             'Has no impact on synthetic anomalies or outlier exposure.'
    )
    
    # 
    # ===================================== heatmap generation =====================================
    # 
    parser.add_argument(
        '--blur-heatmaps', dest='blur_heatmaps', action='store_true',
        help='Blurs heatmaps, like done for the explanation baseline experiments in the paper.'
    )
    parser.add_argument(
        '--gauss-std', type=float, default=10,
        help='Sets a constant value for the standard deviation of the Gaussian kernel used for upsampling and '
             'blurring.'
    )
    parser.add_argument(
        '--quantile', type=float, default=0.97,
        help='The quantile that is used to normalize the generated heatmap images. '
             'This is explained in the Appendix of the paper.'
    )
    parser.add_argument(
        '--resdown', type=int, default=64,
        help='Sets the maximum resolution of logged images (per heatmap), images will be downsampled '
             'if they exceed this threshold. For instance, resdown=64 makes every image of heatmaps contain '
             'individual heatmaps and inputs of width 64 and height 64 at most.'
    )
    parser.add_argument(
        '--no-test', dest="test", action="store_false",
        help='If set then the model will not be tested at the end of the training. It will by default.'
    )
    parser.add_argument(
        "--pixel-level-loss", dest="pixel_level_loss", action="store_true",
        help="If set, the pixel-level loss is used instead of the old version, which didn't apply the anomalous part of the loss to each pixel individually. "
    )
    
    # 
    # ===================================== directories and logging =====================================
    # 
    parser.add_argument(
        '--logdir', type=str, default=pt.join('..', '..', 'data', 'results', 'fcdd_{t}'),
        help='Directory where log data is to be stored. The pattern {t} is replaced by the start time. '
             'Defaults to ../../data/results/fcdd_{t}. '
    )
    parser.add_argument(
        '--logdir-suffix', type=str, default='',
        help='String suffix for log directory, again {t} is replaced by the start time. '
    )
    parser.add_argument(
        '--datadir', type=str, default=pt.join('..', '..', 'data', 'datasets'),
        help='Directory where datasets are found or to be downloaded to. Defaults to ../../data/datasets.',
    )
    
    # 
    # ===================================== dataset =====================================
    # 
    parser.add_argument('--dataset', type=str, default='custom', choices=DATASET_CHOICES)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--nworkers', type=int, default=2)
    parser.add_argument('--pin-memory', action='store_true')
    parser.add_argument(
        '--preproc', type=str, default=PREPROCESSING_DEFAULT_DEFAULT, choices=PREPROCESSING_CHOICES,
        help='Determines the kind of preprocessing pipeline (augmentations and such). '
             'Have a look at the code (dataset implementation, e.g. fcdd.datasets.cifar.py) for details.'
    )
    
    # 
    # ===================================== wandb =====================================
    # 
    parser.add_argument(
        "--wandb-project", type=str, default=None,
        help="If set, the model will be logged to wandb with the project name given here."
    )
    parser.add_argument(
        "--wandb-tags", type=str, nargs='*', default=None,
        help="If set, the model will be logged to wandb with the given tags.",
    )
    parser.add_argument(
        "--wandb-profile", action="store_true",
        help="If set, the run will be profiled and sent to wandb."
    )
    parser.add_argument(
        "--wandb-offline", action="store_true",
        help="If set, the model will be logged to wandb without the need for an internet connection.",
    )
    
    # 
    # ===================================== pytorch lightning =====================================
    # 
    parser.add_argument(
        "--lightning-accelerator", type=str, default=LIGHTNING_ACCELERATOR_GPU, choices=LIGHTNING_ACCELERATOR_CHOICES,
    )
    parser.add_argument(
        "--lightning-ndevices", type=int, default=1,
    )
    return parser


def default_parser_config_mvtec(parser: ArgumentParser) -> ArgumentParser:
    
    parser.set_defaults(
        supervise_mode=mvtec_datamodule.SUPERVISE_MODE_SYNTHETIC_ANOMALY_CONFETTI,
        weight_decay=1e-4, 
        epochs=50, 
        net='FCDD_CNN224_VGG', 
        # anomaly settings
        # heatmap generation
        quantile=0.99, 
        gauss_std=12, 
        # directories and logging
        # dataset
        preproc=mvtec_datamodule.PREPROCESSING_LCNAUG1,
        dataset=DATASET_MVTEC,
        batch_size=128, 
        nworkers=2,
        pin_memory=False,
        # wandb 
        # pytorch lightning 
        lightning_accelerator=LIGHTNING_ACCELERATOR_GPU,
        lightning_ndevices=1,
    )

    parser.add_argument(
        '--it', type=int, default=5, 
        help='Number of runs per class with different random seeds.')
    parser.add_argument(
        '--cls-restrictions', type=int, nargs='+', default=None,
        help='Run only training sessions for some of the classes being nominal.'
    )
    return parser


# In[4]:


def time_format(i: float) -> str:
    """ takes a timestamp (seconds since epoch) and transforms that into a datetime string representation """
    return datetime.fromtimestamp(i).strftime('%Y%m%d%H%M%S')


def args_post_parse(args_):
    
    args_.logdir = Path(args_.logdir)
    logdir_name = args_.logdir.name
    
    # it is duplicated for compatibility with setup_trainer
    args_.log_start_time = int(time.time())
    args_.log_start_time_str = time_format(args_.log_start_time)
    
    logdir_name = f"{args_.dataset}_" + logdir_name
    
    if 'logdir_suffix' in vars(args_):
        logdir_name += args_.logdir_suffix

        del vars(args_)['logdir_suffix']
        
    logdir_name = logdir_name.replace('{t}', args_.log_start_time_str)
    
    args_.logdir = args_.logdir.parent / logdir_name
            
    return args_


# In[]:
# # setup


# In[5]:


TrainSetup = namedtuple(
    "TrainSetup",
    [
        "net",
        "opt",
        "sched",
        "logger",
        "device",
        "quantile",
        "resdown",
        "blur_heatmaps",
    ]
)



# In[]:

import pytorch_lightning as pl

# # training
class TorchTensorboardProfilerCallback(pl.Callback):
  """Quick-and-dirty Callback for invoking TensorboardProfiler during training.
  
  For greater robustness, extend the pl.profiler.profilers.BaseProfiler. See
  https://pytorch-lightning.readthedocs.io/en/stable/advanced/profiler.html"""

  def __init__(self, profiler):
    super().__init__()
    self.profiler = profiler 

  def on_train_batch_end(self, trainer, pl_module, outputs, *args, **kwargs):
    self.profiler.step()
    pl_module.log_dict(outputs)  # also logging the loss, while we're here

# In[]:

def run_one(
    # net
    net: str, 
    bias: bool,
    gauss_std: float,
    pixel_level_loss: bool,
    # optimizer
    optimizer: str, 
    learning_rate: float, 
    weight_decay: float, 
    # scheduler
    scheduler: str,
    lr_sched_param: list,
    # else
    config: str, 
    it: int, 
    epochs: int,
    test: bool,
    normal_class_label: str,
    wandb_logger,
    supervise_mode: str,
    dataset: str,
    preproc: str,
    datadir: str,
    normal_class: int,
    batch_size: int,
    real_anomaly_limit: int,
    logdir: str,
    log_start_time,
    #
    nworkers: int, 
    pin_memory: bool,
    seed: int,
    #
    **kwargs,
):
    """
    kwargs should contain all parameters of the setup function in training.setup
    """

    assert dataset in DATASET_CHOICES, f"Invalid dataset: {dataset}, chose from {DATASET_CHOICES}"
    
    assert supervise_mode in SUPERVISE_MODE_CHOICES, f"Invalid supervise_mode: {supervise_mode}, chose from {SUPERVISE_MODE_CHOICES}"
    assert supervise_mode in dataset_supervise_mode_choices(dataset), f"Invalid supervise_mode: {supervise_mode}, chose from {dataset_supervise_mode_choices(dataset)}"
    
    assert preproc in PREPROCESSING_CHOICES, f"Invalid preproc: {preproc}, chose from {PREPROCESSING_CHOICES}"
    assert preproc in dataset_preprocessing_choices(dataset), f"Invalid preproc: {preproc}, chose from {dataset_preprocessing_choices(dataset)}"
    
    logger = Logger(logdir=logdir, exp_start_time=log_start_time,)
    
    datamodule = MVTecAnomalyDetectionDataModule(
        root=datadir,
        normal_class=normal_class,
        preproc=preproc,
        supervise_mode=supervise_mode,
        real_anomaly_limit=real_anomaly_limit,
        raw_shape=(1, 240, 240),
        batch_size=batch_size,
        nworkers=nworkers,
        pin_memory=pin_memory,
        seed=seed,
    )
            
    # generate preview
    datamodule.prepare_data()
    datamodule.setup()
    images = datamodule.torchvision_dataset.preview(percls=20, train=True)
    logger.imsave(
        name='ds_preview', 
        tensors=torch.cat([*images]), 
        nrow=images.size(1),
        rowheaders=['norm', 'anom', '', 'gtno', 'gtan',],
    )   
    
    # ================================ NET & OPTIMIZER ================================
    try:
        # ds.shape: of the inputs the model expects (n x c x h x w).
        net = MODEL_CLASSES[net](
            
            # model
            in_shape=datamodule.net_shape, 
            gauss_std=gauss_std,
            bias=bias, 
            pixel_level_loss=pixel_level_loss,
            
            # optimizer
            optimizer_name=optimizer,
            lr=learning_rate,
            weight_decay=weight_decay,
            
            # scheduler
            scheduler_name=scheduler,
            lr_sched_param=lr_sched_param,
            
            # else
            normal_class_label=normal_class_label,
        )
    
    except KeyError as err:
        raise KeyError(f'Model {net} is not implemented!') from err

    logger.save_params(net, config)
    
    # Set up profiler
    wait, warmup, active, repeat = 1, 1, 2, 1
    schedule =  torch.profiler.schedule(
      wait=wait, warmup=warmup, active=active, repeat=repeat,
    )
    profile_dir = Path(f"{wandb_logger.save_dir}/latest-run/tbprofile").absolute()
    profiler = torch.profiler.profile(
      schedule=schedule, 
      on_trace_ready=tensorboard_trace_handler(str(profile_dir)), 
      with_stack=True,
    )

    with profiler:
        
        profiler_callback = TorchTensorboardProfilerCallback(profiler)
        
        trainer = pl.Trainer(
            logger=wandb_logger,  
            log_every_n_steps=1,  
            gpus=1, 
            max_epochs=epochs,    
            deterministic=True,
            callbacks=[profiler_callback],   
        )
        trainer.fit(model=net, datamodule=datamodule)

    profile_art = wandb.Artifact(f"trace-{wandb.run.id}", type="profile")
    profile_art.add_file(str(next(profile_dir.glob("*.pt.trace.json"))), "trace.pt.trace.json")
    wandb_logger.experiment.log_artifact(profile_art)

    if not test:
        return 
    
    trainer.test(model=net, datamodule=datamodule)    
    
    
def run(**kwargs) -> dict:
    
    base_logdir = kwargs['logdir']
    dataset = kwargs['dataset']
    
    cls_restrictions = kwargs.pop("cls_restrictions", None)
    classes = cls_restrictions or range(dataset_nclasses(dataset))

    number_it = kwargs.pop('it')
    seeds = kwargs.pop('seeds')
    
    if seeds is None:
        assert number_it is not None, "seeds or number_it must be specified"
        print('no seeds specified, using default: auto generated seeds from the system entropy')
        its = list(range(number_it))
        seeds = []
        for _ in its:
            seeds.append(create_seed())
            time.sleep(1/3)  # let the system state change
    else:
        if number_it is None:
            number_it = len(seeds)
            print(f"number_it is not specified, using {number_it} (from the seeds given)")
        assert len(seeds) == number_it, f"number_it {number_it} != len(seeds) {len(seeds)}"
        for s in seeds:
            assert type(s) == int, f"seed must be an int, got {type(s)}"
            assert s >= 0, f"seed must be >= 0, got {s}"
        assert len(set(seeds)) == len(seeds), f"seeds must be unique, got {s}"
        
    its = list(range(len(seeds)))
        
    results = []
    
    for c in classes:
        cls_logdir = base_logdir / f'normal_{c}'
        
        kwargs['normal_class'] = c
        kwargs['normal_class_label'] = dataset_class_labels(dataset)[c]
    
        for i, seed in zip(its, seeds):
            logdir = (cls_logdir / 'it_{:02}'.format(i)).absolute()
            
            wandb_project = kwargs.pop("wandb_project", None)
            wandb_tags = kwargs.pop("wandb_tags", None) or []
            wandb_profile = kwargs.pop("wandb_profile", False)
            wandb_dir = str(logdir / 'wandb')
            wandb_offline = kwargs.pop("wandb_offline", False)
            
            # wandb_init = wandb.init if wandb_project is not None else no_wandb_init 
            del kwargs["log_start_time_str"]
            
            kwargs = {
                **kwargs, 
                **dict(
                    logdir=str(logdir),  # overwrite logdir
                    it=i,
                    datadir=str(Path(kwargs["datadir"]).absolute()),
                    seed=seed,
                )
            }
            
            torch.manual_seed(seed)
            
            wandb_logger = WandbLogger(
                project=wandb_project, 
                name=f"{logdir.parent.parent.name}.{logdir.parent.name}.{logdir.name}",
                entity="mines-paristech-cmm",
                save_dir=wandb_dir,
                offline=wandb_offline,
                # id=wandb_id,  # to restart from a previous run
                # kwargs
                tags=wandb_tags,
                config=kwargs,
                # save_code=True,
            )
            
            kwargs["config"] = json.dumps(kwargs)
            run_one(
                wandb_logger=wandb_logger,
                **kwargs,
            )
            
            results.append(dict(class_idx=c, it=i, logdir=str(logdir)))

    return results


# %%

# In[]:
# # launch

# In[22]:

# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
# os.environ["WANDB"] = "1"
# os.environ["TMPDIR"] = "/data/bertoldo/tmp"

# ARG_STRING = "--cls-restrictions 0 --it 1 --epochs 3"

# args = ARG_STRING.split(" ")

if __name__ == "__main__":
    
    parser = ArgumentParser(
        description="""
        Train a neural network module as explained in the `Explainable Deep Anomaly Detection` paper.
        Train FCDD, and log achieved scores, metrics, plots, and heatmaps
        for both test and training data. 
        """
    )
    parser = default_parser_config(parser)
    parser = default_parser_config_mvtec(parser)
    args = parser.parse_args()
    args = args_post_parse(args)
    results = run(**vars(args))


