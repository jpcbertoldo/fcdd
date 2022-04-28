#!/usr/bin/env python
# coding: utf-8

# In[]:
# # train mvtec
# 
# i want to have a simpler script to integrate to wandb and later adapt it to unetdd

# In[]:
# # imports

# In[1]:

import collections
import glob
import json
import os
import os.path as pt
import random
import time
import traceback
from argparse import ArgumentParser
from collections import namedtuple
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Union
from dev.train_mvtec_dev01_aux import single_save

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchmetrics
import torchvision
from fcdd.models.bases import BaseNet, ReceptiveNet
from fcdd.util.logging import Logger
from fcdd.util.logging import colorize as colorize_img
from kornia import gaussian_blur2d
from pytorch_lightning.loggers import WandbLogger
from scipy.interpolate import interp1d
from sklearn.metrics import (auc, average_precision_score,
                             precision_recall_curve, roc_curve)
from torch import Tensor, gt
from torch.hub import load_state_dict_from_url
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset

import wandb

from mvtec_dataset import ADMvTec
from train_mvtec_dev01_aux import compute_gtmap_roc, compute_gtmap_pr

from torch.profiler import tensorboard_trace_handler

# from fcdd.datasets.noise import kernel_size_to_std
# from fcdd.training import balance_labels
# from fcdd.training.setup import pick_opt_sched
# from fcdd.models import choices, load_nets

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
# # consts

SUPERVISE_MODES = (
    # 'unsupervised', 
    # 'other', 
    'noise', 
    'malformed_normal', 
    'malformed_normal_gt'
)
NOISE_MODES = [
    # Synthetic Anomalies
    'confetti',  
    # Outlier Exposure online supervision only  
    'mvtec', 
    'mvtec_gt'  
]

# In[]:
# # datasets

DATASET_CHOICES = ('mvtec',)


def dataset_class_labels(dataset_name: str) -> List[str]:
    return {
        'mvtec': deepcopy(ADMvTec.classes_labels),
    }[dataset_name]


def dataset_nclasses(dataset_name: str) -> int:
    return len(dataset_class_labels(dataset_name))


def dataset_class_index(dataset_name: str, class_name: str) -> int:

    return dataset_class_labels(dataset_name).index(class_name)


def dataset_preprocessing_choices(dataset_name: str) -> List[str]:
    return {
        'mvtec': deepcopy(ADMvTec.preprocessing_choices),
    }[dataset_name]


PREPROCESSING_CHOICES = tuple(set.union(*[
    set(dataset_preprocessing_choices(dataset_name)) 
    for dataset_name in DATASET_CHOICES
]))

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
    
    def loss(self, inputs: Tensor, gtmaps: Tensor, labels: Tensor) -> Tensor:
        """ computes the FCDD """
        
        batch_size, *dims = inputs.size()
        
        anomaly_scores = self(inputs) 
        
        loss_map = anomaly_scores ** 2
        loss_map = (loss_map + 1).sqrt() - 1
        
        gauss_std = self.hparams["gauss_std"]
        loss_map = self._receptive_field_net.receptive_upsample(loss_map, reception=True, std=gauss_std, cpu=False)
        
        norm_map = (loss_map * (1 - gtmaps))
        norm = norm_map.view(batch_size, -1).mean(-1)
        
        # this is my implementation
        pixel_level_loss = self.hparams["pixel_level_loss"]
        if pixel_level_loss:
            anom_map = -(((1 - (-loss_map).exp()) + 1e-31).log())
            anom_map = anom_map * gtmaps
            anom = anom_map.view(batch_size, -1).mean(-1)

        # this is fcdd's implementation
        else:            
            anom_map = torch.zeros_like(norm_map)
            exclude_complete_nominal_samples = ((gtmaps == 1).view(gtmaps.size(0), -1).sum(-1) > 0)
            anom = torch.zeros_like(norm)
            if exclude_complete_nominal_samples.sum() > 0:
                a = (loss_map * gtmaps)[exclude_complete_nominal_samples]
                anom[exclude_complete_nominal_samples] = (
                    -(((1 - (-a.view(a.size(0), -1).mean(-1)).exp()) + 1e-31).log())
                )
        
        loss_map = norm_map + anom_map
        loss = loss_map.mean()
        
        return anomaly_scores, loss_map, loss
    
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
        anomaly_scores_maps, loss_maps, loss = self.loss(inputs=inputs, gtmaps=gtmaps, labels=labels)
        
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
        anomaly_scores_maps, loss_maps, loss = self.loss(inputs=inputs, gtmaps=gtmaps, labels=labels)
        
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

    # define directories for datasets and logging
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

    # training parameters
    parser.add_argument('-b', '--batch-size', type=int, default=128)
    parser.add_argument('-e', '--epochs', type=int, default=200)
    parser.add_argument('-w', '--workers', type=int, default=4)
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
    parser.add_argument('-d', '--dataset', type=str, default='custom', choices=DATASET_CHOICES)
    parser.add_argument(
        '-n', '--net', type=str, default='FCDD_CNN224_VGG', choices=MODEL_CLASSES.keys(),
        help='Chooses a network architecture to train.'
    )
    parser.add_argument(
        '--preproc', type=str, default='aug1', choices=PREPROCESSING_CHOICES,
        help='Determines the kind of preprocessing pipeline (augmentations and such). '
             'Have a look at the code (dataset implementation, e.g. fcdd.datasets.cifar.py) for details.'
    )
    parser.add_argument('--no-bias', dest='bias', action='store_false', help='Uses no bias in network layers.')
    parser.add_argument('--cpu', dest='cuda', action='store_false', help='Trains on CPU only.')

    # artificial anomaly settings
    parser.add_argument(
        '--supervise-mode', type=str, default='noise', choices=SUPERVISE_MODES,
        help='This determines the kind of artificial anomalies. '
             '"unsupervised" uses no anomalies at all. '
             '"other" uses ground-truth anomalies. '
             '"noise" uses pure noise images or Outlier Exposure. '
             '"malformed_normal" adds noise to nominal images to create malformed nominal anomalies. '
             '"malformed_normal_gt" is like malformed_normal, but with ground-truth anomaly heatmaps for training. '
    )
    parser.add_argument(
        '--noise-mode', type=str, default='imagenet22k', choices=NOISE_MODES,
        help='The type of noise used when artificial anomalies are activated. Dataset names refer to OE. '
             'See fcdd.datasets.noise_modes.py.'
    )
    parser.add_argument(
        '--oe-limit', type=int, default=np.infty,
        help='Determines the amount of different samples used for Outlier Exposure. '
             'Has no impact on synthetic anomalies.'
    )
    parser.add_argument(
        '--nominal-label', type=int, default=0,
        help='Determines the label that marks nominal samples. '
             'Note that this is not the class that is considered nominal! '
             'For instance, class 5 is the nominal class, which is labeled with the nominal label 0.'
    )

    # heatmap generation parameters
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
    return parser


def default_parser_config_mvtec(parser: ArgumentParser) -> ArgumentParser:
    
    parser.set_defaults(
        batch_size=16, 
        supervise_mode='malformed_normal',
        gauss_std=12, 
        weight_decay=1e-4, 
        epochs=200, 
        preproc='lcnaug1',
        quantile=0.99, 
        net='FCDD_CNN224_VGG', 
        dataset='mvtec', 
        noise_mode='confetti',
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
        "dataset",
        "train_loader",
        "test_loader",
        "opt",
        "sched",
        "logger",
        "device",
        "quantile",
        "resdown",
        "blur_heatmaps",
    ]
)

def trainer_setup(
    dataset: str, 
    datadir: str, 
    logdir: str, 
    batch_size: int,
    preproc: str, 
    supervise_mode: str, 
    nominal_label: int,
    oe_limit: int, 
    noise_mode: str,
    workers: int, 
    quantile: float, 
    resdown: int, 
    blur_heatmaps: bool,
    cuda: bool, 
    log_start_time: int = None, 
    normal_class: int = 0,
) -> TrainSetup:
    """
    Creates a complete setup for training, given all necessary parameter from a runner (seefcdd.runners.bases.py).
    This includes loading networks, datasets, data loaders, optimizers, and learning rate schedulers.
    :param dataset: dataset identifier string (see :data:`fcdd.datasets.DS_CHOICES`).
    :param datadir: directory where the datasets are found or to be downloaded to.
    :param logdir: directory where log data is to be stored.
    :param net: network model identifier string (see :func:`fcdd.models.choices`).
    :param bias: whether to use bias in the network layers.
    :param learning_rate: initial learning rate.
    :param weight_decay: weight decay (L2 penalty) regularizer.
    :param lr_sched_param: learning rate scheduler parameters. Format depends on the scheduler type.
        For 'milestones' needs to have at least two elements, the first corresponding to the factor
        the learning rate is decreased by at each milestone, the rest corresponding to milestones (epochs).
        For 'lambda' needs to have exactly one element, i.e. the factor the learning rate is decreased by
        at each epoch.
    :param batch_size: batch size, i.e. number of data samples that are returned per iteration of the data loader.
    :param optimizer: optimizer type, needs to be one of {'sgd', 'adam'}.
    :param scheduler: learning rate scheduler type, needs to be one of {'lambda', 'milestones'}.
    :param preproc: data preprocessing pipeline identifier string (see :data:`PREPROCESSING_CHOICES`).
    :param supervise_mode: the type of generated artificial anomalies.
        See :meth:`fcdd.datasets.bases.TorchvisionDataset._generate_artificial_anomalies_train_set`.
    :param nominal_label: the label that is to be returned to mark nominal samples.
    :param oe_limit: limits the number of different anomalies in case of Outlier Exposure (defined in noise_mode).
    :param noise_mode: the type of noise used, see :mod:`fcdd.datasets.noise_mode`.
    :param workers: how many subprocesses to use for data loading.
    :param quantile: the quantile that is used to normalize the generated heatmap images.
    :param resdown: the maximum resolution of logged images, images will be downsampled if necessary.
    :param gauss_std: a constant value for the standard deviation of the Gaussian kernel used for upsampling and
        blurring, the default value is determined by :func:`fcdd.datasets.noise.kernel_size_to_std`.
    :param blur_heatmaps: whether to blur heatmaps.
    :param cuda: whether to use GPU.
    :param config: some config text that is to be stored in the config.txt file.
    :param log_start_time: the start time of the experiment.
    :param normal_class: the class that is to be considered nominal.
    :return: a dictionary containing all necessary parameters to be passed to a Trainer instance.
    """
    assert supervise_mode in SUPERVISE_MODES, 'unknown supervise mode: {}'.format(supervise_mode)
    assert noise_mode in NOISE_MODES, 'unknown noise mode: {}'.format(noise_mode)
    assert dataset in DATASET_CHOICES
    assert preproc in PREPROCESSING_CHOICES
    device = torch.device('cuda:0') if cuda else torch.device('cpu')
    
    logger = Logger(logdir=logdir, exp_start_time=log_start_time,)
    
    # ================================ DATASET ================================
    if dataset == 'mvtec':
        ds = ADMvTec(
            root=datadir, 
            normal_class=normal_class, 
            preproc=preproc,
            supervise_mode=supervise_mode, 
            noise_mode=noise_mode, 
            oe_limit=oe_limit, 
            logger=logger, 
            nominal_label=nominal_label,
        )
    else:
        raise NotImplementedError(f'Dataset {dataset} is unknown.')
    
    train_loader, test_loader = ds.loaders(batch_size=batch_size, num_workers=workers)
       
    # ================================ ELSE ================================
    
    if not hasattr(ds, 'nominal_label') or ds.nominal_label < ds.anomalous_label:
        ds_order = ['norm', 'anom']
    else:
        ds_order = ['anom', 'norm']
        
    images = ds.preview(percls=20, train=True)
    
    rowheaders = [
        *ds_order, 
        '', 
        *['gtno' if s == 'norm' else 'gtan' for s in ds_order]
    ]
        
    logger.imsave(
        name='ds_preview', 
        tensors=torch.cat([*images]), 
        nrow=images.size(1),
        rowheaders=rowheaders,
    )
    
    return TrainSetup(
        net=None, 
        dataset=ds,
        train_loader=train_loader,
        test_loader=test_loader,
        opt=None, 
        sched=None, 
        logger=logger,
        device=device, 
        quantile=quantile, 
        resdown=resdown,
        blur_heatmaps=blur_heatmaps,
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
    # 
    **kwargs,
):
    """
    kwargs should contain all parameters of the setup function in training.setup
    """
    
    setup: TrainSetup = trainer_setup(**kwargs)
    
    # ================================ NET & OPTIMIZER ================================
    try:
        # ds.shape: of the inputs the model expects (n x c x h x w).
        net = MODEL_CLASSES[net](
            
            # model
            in_shape=setup.dataset.shape, 
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

    setup.logger.save_params(net, config)
    
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
        trainer.fit(model=net, train_dataloaders=setup.train_loader)

    profile_art = wandb.Artifact(f"trace-{wandb.run.id}", type="profile")
    profile_art.add_file(str(next(profile_dir.glob("*.pt.trace.json"))), "trace.pt.trace.json")
    wandb_logger.experiment.log_artifact(profile_art)

    if not test:
        return 
    
    trainer.test(model=net, dataloaders=setup.test_loader, ckpt_path=None)    
    
    
def run(**kwargs) -> dict:
    
    base_logdir = kwargs['logdir']
    dataset = kwargs['dataset']
    
    cls_restrictions = kwargs.pop("cls_restrictions", None)
    classes = cls_restrictions or range(dataset_nclasses(dataset))

    number_it = kwargs.pop('it')
    its_restrictions = kwargs.pop("its_restrictions", None)
    its = its_restrictions or range(number_it)

    results = []
    
    for c in classes:
        cls_logdir = base_logdir / f'normal_{c}'
        
        kwargs['normal_class'] = c
        kwargs['normal_class_label'] = dataset_class_labels(dataset)[c]
    
        for i in its:
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
                )
            }
            
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


