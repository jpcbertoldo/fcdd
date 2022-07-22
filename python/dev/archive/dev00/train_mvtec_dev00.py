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
from typing import List, Tuple

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


# from fcdd.datasets.noise import kernel_size_to_std
# from fcdd.training import balance_labels
# from fcdd.training.setup import pick_opt_sched
# from fcdd.models import choices, load_nets

# # utils

def kernel_size_to_std(k: int):
    """ Returns a standard deviation value for a Gaussian kernel based on its size """
    return np.log10(0.45*k + 1) + 0.25 if k < 32 else 10


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

# reduce the number of points of the ROC curve
ROC_PR_CURVES_LIMIT_NUMBER_OF_POINTS = 3000
ROC_PR_CURVES_INTERPOLATION_NUMBER_OF_POINTS = 3000
    
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


class FCDDNet(ReceptiveNet):
    """ Baseclass for FCDD networks, i.e. network without fully connected layers that have a spatial output """
    
    def __init__(self, in_shape: Tuple[int, int, int], gauss_std: float, bias=False, **kwargs):
        super().__init__(in_shape, bias)
        self.gauss_std = float(gauss_std)
    
    def anomaly_score(self, loss: Tensor) -> Tensor:
        """ This assumes the loss is already the anomaly score. If this is not the case, reimplement the method! """
        return loss
    
    def reduce_ascore(self, ascore: Tensor) -> Tensor:
        """ Reduces the anomaly score to be a score per image (detection). """
        return ascore.reshape(ascore.size(0), -1).mean(1)
    
    def loss(self, outs: Tensor, ins: Tensor, labels: Tensor, gtmaps: Tensor = None, pixel_level_loss: bool = False) -> Tensor:
        """ computes the FCDD """
        loss = outs ** 2
        loss = (loss + 1).sqrt() - 1
        
        if self.training:
            
            assert gtmaps is not None
            
            loss = self.receptive_upsample(loss, reception=True, std=self.gauss_std, cpu=False)
            
            norm_map = (loss * (1 - gtmaps))
            norm = norm_map.view(norm_map.size(0), -1).mean(-1)
            
            # this is my implementation
            if pixel_level_loss:
                anom_map = -(((1 - (-loss).exp()) + 1e-31).log())
                anom_map = anom_map * gtmaps
                anom = anom_map.view(anom_map.size(0), -1).mean(-1)

            # this is fcdd's implementation
            else:            
                exclude_complete_nominal_samples = ((gtmaps == 1).view(gtmaps.size(0), -1).sum(-1) > 0)
                anom = torch.zeros_like(norm)
                if exclude_complete_nominal_samples.sum() > 0:
                    a = (loss * gtmaps)[exclude_complete_nominal_samples]
                    anom[exclude_complete_nominal_samples] = (
                        -(((1 - (-a.view(a.size(0), -1).mean(-1)).exp()) + 1e-31).log())
                    )
               
            return norm + anom
            
        else:
            return loss  # here it is always loss map

class FCDD_CNN224_VGG(FCDDNet):
    """
    # VGG_11BN based net with most of the VGG layers having weights 
    # pretrained on the ImageNet classification task.
    # these weights get frozen, i.e., the weights will not get updated during training
    """

    def __init__(self, in_shape, **kwargs):
        
        super().__init__(in_shape, **kwargs)
        assert self.bias, 'VGG net is only supported with bias atm!'
        
        state_dict = load_state_dict_from_url(
            torchvision.models.vgg.model_urls['vgg11_bn'],
            model_dir=pt.join(pt.dirname(__file__), '..', '..', '..', 'data', 'models')
        )
        features_state_dict = {k[9:]: v for k, v in state_dict.items() if k.startswith('features')}

        self.features = nn.Sequential(
            self._create_conv2d(3, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            self._create_maxpool2d(2, 2),
            self._create_conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            self._create_maxpool2d(2, 2),
            self._create_conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            self._create_conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            self._create_maxpool2d(2, 2),
            # Frozen version freezes up to here
            self._create_conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            self._create_conv2d(512, 512, 3, 1, 1),
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

        self.conv_final = self._create_conv2d(512, 1, 1)
        
        for m in self.features[:15]:
            for p in m.parameters():
                p.requires_grad = False

    def forward(self, x, ad=True):
        x = self.features(x)
        if ad:
            x = self.conv_final(x)
        return x
       
   
MODEL_CLASSES = {
    "FCDD_CNN224_VGG": FCDD_CNN224_VGG,
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
    parser.add_argument(
        '--readme', type=str, default='',
        help='Some notes to be stored in the automatically created config.txt configuration file.'
    )

    # training parameters
    parser.add_argument('-b', '--batch-size', type=int, default=128)
    parser.add_argument('-e', '--epochs', type=int, default=200)
    parser.add_argument('-w', '--workers', type=int, default=4)
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3)
    parser.add_argument('-wd', '--weight-decay', type=float, default=1e-6)
    parser.add_argument(
        '--optimizer-type', type=str, default='sgd', choices=['sgd', 'adam'],
        help='The type of optimizer. Defaults to "sgd". '
    )
    parser.add_argument(
        '--scheduler-type', type=str, default='lambda', choices=['lambda', 'milestones'],
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
        '--load', type=str, default=None,
        help='Path to a file that contains a snapshot of the network model. '
             'When given, the network loads the found weights and state of the training. '
             'If epochs are left to be trained, the training is continued. '
             'Note that only one snapshot is given, thus using a runner that trains for multiple different classes '
             'to be nominal is not applicable. '
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
    parser.add_argument(
        '--acc-batches', type=int, default=1,
        help='To speed up data loading, '
             'this determines the number of batches that are accumulated to be used for training. '
             'For instance, acc_batches=2 iterates the data loader two times, concatenates the batches, and '
             'passes the result to the further training procedure. This has no impact on the performance '
             'if the batch size is reduced accordingly (e.g. one half in this example), '
             'but can decrease training time. '
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
    return parser


def default_parser_config_mvtec(parser: ArgumentParser) -> ArgumentParser:
    
    parser.set_defaults(
        batch_size=16, 
        acc_batches=8, 
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
        "gauss_std",
        "blur_heatmaps",
        "pixel_level_loss",
    ]
)

def trainer_setup(
    dataset: str, 
    datadir: str, 
    logdir: str, 
    net: str, 
    bias: bool,
    learning_rate: float, 
    weight_decay: float, 
    lr_sched_param: List[float], 
    batch_size: int,
    optimizer_type: str, 
    scheduler_type: str,
    preproc: str, 
    supervise_mode: str, 
    nominal_label: int,
    oe_limit: int, 
    noise_mode: str,
    workers: int, 
    quantile: float, 
    resdown: int, 
    gauss_std: float, 
    blur_heatmaps: bool,
    cuda: bool, 
    config: str, 
    pixel_level_loss: bool,
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
    :param optimizer_type: optimizer type, needs to be one of {'sgd', 'adam'}.
    :param scheduler_type: learning rate scheduler type, needs to be one of {'lambda', 'milestones'}.
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
   
    # ================================ NET ================================
    # net = load_nets(name=net, in_shape=ds.shape, bias=bias)
    
    try:
        # ds.shape: of the inputs the model expects (n x c x h x w).
        net = MODEL_CLASSES[net](ds.shape, bias=bias, gauss_std=gauss_std).to(device)
    
    except KeyError:
        raise KeyError(f'Model {net} is not implemented!')  
        
    # ================================ OPTIMIZER ================================
    if optimizer_type == 'sgd':
        optimizer = optim.SGD(net.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=0.9, nesterov=True)
    
    elif optimizer_type == 'adam':
        optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    else:
        raise NotImplementedError('Optimizer type {} not known.'.format(optimizer_type))
    
    # ================================ SCHEDULER ================================
    if scheduler_type == 'lambda':
        assert len(lr_sched_param) == 1 and 0 < lr_sched_param[0] <= 1
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda ep: lr_sched_param[0] ** ep)
    
    elif scheduler_type == 'milestones':
        assert len(lr_sched_param) >= 2 and 0 < lr_sched_param[0] <= 1 and all([p > 1 for p in lr_sched_param[1:]])
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [int(s) for s in lr_sched_param[1:]], lr_sched_param[0], )
    
    else:
        raise NotImplementedError('LR scheduler type {} not known.'.format(scheduler_type))
    
    # ================================ ELSE ================================
    
    logger.save_params(net, config)
    
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
        net=net, 
        dataset=ds,
        train_loader=train_loader,
        test_loader=test_loader,
        opt=optimizer, 
        sched=scheduler, 
        logger=logger,
        device=device, 
        quantile=quantile, 
        resdown=resdown,
        gauss_std=gauss_std, 
        blur_heatmaps=blur_heatmaps,
        pixel_level_loss=pixel_level_loss,
    )
    
def train_setup_save_snapshot(outfile: str, net: FCDDNet = None, opt=None, sched=None, epoch: int = None, **kwargs):
    """
    Saves a snapshot of the current state of the training setup.
    Writes a snapshot of the training, i.e. network weights, optimizer state and scheduler state to a file
    in the log directory. 
    All other things in the kwargs are saved as well.
    :param setup: the training setup.
    :return:
    """
    if not pt.exists(os.path.dirname(outfile)):
        os.makedirs(os.path.dirname(outfile))
        
    torch.save(
        {
            'net': net.state_dict(), 
            'opt': opt.state_dict(), 
            'sched': sched.state_dict(), 
            'epoch': epoch,
            'kwargs': kwargs,
        }
        , outfile
    )



def train_setup_load_snapshot(path: str, device, net: FCDDNet = None, opt=None, sched=None) -> int:
    """ Loads a snapshot of the training state, including network weights """
    
    raise NotImplementedError("this method has not been tested yet")
    
    snapshot = torch.load(path, map_location=device)
    
    net_state = snapshot.pop('net', None)
    opt_state = snapshot.pop('opt', None)
    sched_state = snapshot.pop('sched', None)
    epoch = snapshot.pop('epoch', None)
    kwargs = snapshot.pop('kwargs', None)
    
    if net_state is not None and net is not None:
        net.load_state_dict(net_state)
        
    if opt_state is not None and opt is not None:
        opt.load_state_dict(opt_state)
        
    if sched_state is not None and sched is not None:
        sched.load_state_dict(sched_state)
        
    print(
        "Loaded {}{}{} with starting epoch {}".format(
        'net_state, ' if net_state else '', 'opt_state, ' if opt_state else '',
        'sched_state' if sched_state else '', epoch
    ))
    return epoch, kwargs


# In[]:
# # trainer

# In[13]:

class FCDDTrainer:
    
    def __init__(
        self, 
        net: BaseNet, 
        logger: Logger, 
        gauss_std: float, 
        quantile: float, 
        resdown: int, 
        blur_heatmaps=False,
        device='cuda:0',
        pixel_level_loss: bool = False, 
        pixel_loss_fix: bool = False,
        **kwargs
    ):
        """
        The train method is modified to be able to handle ground-truth maps.
        Anomaly detection trainer that defines a test phase where scores are computed and heatmaps are generated.
        :param net: some neural network instance
        :param opt: optimizer.
        :param sched: learning rate scheduler.
        :param dataset_loaders:
        :param logger: some logger.
        :param device: some torch device, either cpu or gpu.
        :param gauss_std: a constant value for the standard deviation of the Gaussian kernel used for upsampling and
            blurring, the default value is determined by :func:`fcdd.datasets.noise.kernel_size_to_std`.
        :param quantile: the quantile that is used to normalize the generated heatmap images.
        :param resdown: the maximum resolution of logged images, images will be downsampled if necessary.
        :param blur_heatmaps: whether to blur heatmaps.
        :param pixel_level_loss: whether to use pixel-level loss (my implementation) instead of fcdd's, where it is disregarded
        """
        self.net = net
        self.logger = logger
        self.device = device
        self.gauss_std = gauss_std
        self.quantile = quantile
        self.resdown = resdown
        self.blur_heatmaps = blur_heatmaps
        self.pixel_level_loss = pixel_level_loss
        self.pixel_loss_fix = pixel_loss_fix
                
    def train(
        self, 
        net: FCDDNet, 
        opt: Optimizer, 
        sched: _LRScheduler, 
        logger: Logger, 
        train_loader: DataLoader, 
        epochs: int, 
        device,
        acc_batches=1, 
        wandb=None,
    ) -> BaseNet:
        """
        Does epochs many full iteration of the data loader and trains the network with the data using self.loss.
        Supports ground-truth maps, logs losses for
        nominal and anomalous samples separately, and introduces another parameter to
        accumulate batches for faster data loading.
        :param epochs: number of full data loader iterations to train.
        :param acc_batches: To speed up data loading, this determines the number of batches that are accumulated
            before forwarded through the network. For instance, acc_batches=2 iterates the data loader two times,
            concatenates the batches, and passes this to the network. This has no impact on the performance
            if the batch size is reduced accordingly (e.g. one half in this example), but can decrease training time.
        :return: the trained network
        """
        
        assert 0 < acc_batches and isinstance(acc_batches, int)
        
        net = net.to(device).train()
        
        for epoch in range(epochs):
            
            acc_data, acc_counter = [], 1
            
            for n_batch, data in enumerate(train_loader):
                
                if acc_counter < acc_batches and n_batch < len(train_loader) - 1:
                    acc_data.append(data)
                    acc_counter += 1
                    continue
                elif acc_batches > 1:
                    acc_data.append(data)
                    data = [torch.cat(d) for d in zip(*acc_data)]
                    acc_data, acc_counter = [], 1


                inputs, labels, gtmaps = data
                inputs = inputs.to(device)
                gtmaps = gtmaps.to(device)
                opt.zero_grad()
                outputs = net(inputs)
                loss = net.loss(outputs, inputs, labels, gtmaps, pixel_level_loss=self.pixel_level_loss)
                loss_mean = loss.mean()
                loss_mean.backward()
                opt.step()
                with torch.no_grad():
                    info = {}
                    if len(set(labels.tolist())) > 1:
                        swloss = loss.reshape(loss.size(0), -1).mean(-1)
                        info = {'err_normal': swloss[labels == 0].mean(),
                                'err_anomalous': swloss[labels != 0].mean()}
                    logger.log(
                        epoch, 
                        n_batch, 
                        len(train_loader), 
                        loss_mean,
                        infoprint='LR {} ID {}{}'.format(
                            ['{:.0e}'.format(p['lr']) for p in opt.param_groups],
                            str(self.__class__)[8:-2],
                            ' NCLS {}'.format(train_loader.dataset.normal_classes)
                            if hasattr(train_loader.dataset, 'normal_classes') else ''
                        ),
                        info=info
                    )
            if wandb is not None:
                wandb.log(dict(
                    epoch=epoch,
                    epoch_percent=(1 + epoch) / epochs,  # 1+ makes it finish at 100%
                    loss=loss_mean.data.item(),
                ))
            sched.step()

        return net

    def test(
        self, 
        net: FCDDNet, 
        data_loader: DataLoader, 
        logger: Logger, 
        device, 
    ):
        """
        Does a full iteration of a data loader, remembers all data (i.e. inputs, labels, outputs, loss),
        and computes scores and heatmaps with it. 
        
        For each, one heatmap picture is generated that contains (row-wise):
            -   The first 20 nominal samples (label == 0, if nominal_label==1 this shows anomalies instead).
            -   The first 20 anomalous samples (label == 1, if nominal_label==1 this shows nominal samples instead).
                The :func:`reorder` takes care that the first anomalous test samples are not all from the same class.
            -   The 10 most nominal rated samples from the nominal set on the left and
                the 10 most anomalous rated samples from the nominal set on the right.
            -   The 10 most nominal rated samples from the anomalous set on the left and
                the 10 most anomalous  rated samples from the anomalous set on the right.
        
        Four heatmap pictures are generated that show six samples with increasing anomaly score from left to right. 
        I.E.: the leftmost heatmap shows the most nominal rated example and the rightmost sample the most anomalous rated one. 
        There are two heatmaps for the anomalous set and two heatmaps for the nominal set. 
        Both with either local normalization -- i.e. each heatmap is normalized w.r.t itself only, 
                            there is a complete red and complete blue pixel in each heatmap -- or semi-global normalization -- 
        each heatmap is normalized w.r.t. to all heatmaps shown in the picture.
        
        These four heatmap pictures are also stored as tensors in a 'tim' subdirectory for later usage.
        
        The score computes AUC values and complete ROC curves for detection. It also computes explanation ROC curves
        if ground-truth maps are available.

        :param specific_viz_ids: in addition to the heatmaps generated above, this also generates heatmaps
            for specific sample indices. The first element of specific_viz_ids is for nominal samples
            and the second for anomalous ones. The resulting heatmaps are stored in a `specific_viz_ids` subdirectory.
        :return: A dictionary of ROC results, each ROC result is again represented by a dictionary of the form: {
                'tpr': [], 'fpr': [], 'ths': [], 'auc': int, ...
            }.
        """
        net = net.to(device).eval()   # HEREHERE HERE HERE HERE HERE HERE HERE HERE HERE HERE HERE HERE HERE HERE HERE HERE HERE 
        # next: remove heatmap generation from here (? is it worth to break it down?), extract test from the trainer 
        
        logger.print('Testing data split `test`', fps=False)
        
        all_labels, all_loss, all_anomaly_scores, all_imgs, all_outputs = [], [], [], [], []
        all_gtmaps = []
        
        for n_batch, data in enumerate(data_loader):
            
            inputs, labels, gtmaps = data
            all_gtmaps.append(gtmaps)

            # return outputs, loss, anomaly_score, grads
            # outputs, loss, anomaly_score, grads = self._regular_forward(inputs, labels)
            inputs = inputs.to(device)
            with torch.no_grad():
                outputs = net(inputs)
                loss = net.loss(outputs, inputs, labels)
                anomaly_score = net.anomaly_score(loss)
            
            all_labels += labels.detach().cpu().tolist()
            all_loss.append(loss.detach().cpu())
            all_anomaly_scores.append(anomaly_score.detach().cpu())
            all_imgs.append(inputs.detach().cpu())
            all_outputs.append(outputs.detach().cpu())
            
            logger.print(f'TEST {n_batch:04d}/{len(data_loader):04d}', fps=True)
            
        all_imgs = torch.cat(all_imgs)
        all_outputs = torch.cat(all_outputs)
        all_gtmaps = torch.cat(all_gtmaps) if len(all_gtmaps) > 0 else None
        all_loss = torch.cat(all_loss)
        all_anomaly_scores = torch.cat(all_anomaly_scores)
        
        # change variable names because i got rid of a function that did this before
        # labels, loss, anomaly_scores, imgs, outputs, gtmaps, grads = gather_data(data_loader,)
        labels, loss, anomaly_scores, imgs, outputs, gtmaps = all_labels, all_loss, all_anomaly_scores, all_imgs, all_outputs, all_gtmaps
        
        def reorder(
            labels: List[int], 
            loss: Tensor, 
            anomaly_scores: Tensor, 
            imgs: Tensor, 
            outputs: Tensor, 
            gtmaps: Tensor,
            ds: Dataset = None
        ) -> Tuple[List[int], Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
            """ returns all inputs in an identical new order if the dataset offers a predefined (random) order """
            if ds is not None and hasattr(ds, 'fixed_random_order'):
                assert gtmaps is None, 'original gtmaps loaded in score do not know order! Hence reordering is not allowed for GT datasets'
                o = ds.fixed_random_order
                labels = labels[o] if isinstance(labels, (Tensor, np.ndarray)) else np.asarray(labels)[o].tolist()
                loss, anomaly_scores, imgs = loss[o], anomaly_scores[o], imgs[o]
                outputs, gtmaps = outputs[o], gtmaps
            return labels, loss, anomaly_scores, imgs, outputs, gtmaps
        
        labels, loss, anomaly_scores, imgs, outputs, gtmaps = reorder(
            labels=labels,
            loss=loss,
            anomaly_scores=anomaly_scores,
            imgs=imgs,
            outputs=outputs,
            gtmaps=gtmaps,
            ds=data_loader.dataset,
        )
        
        return labels, loss, anomaly_scores, imgs, outputs, gtmaps
    
    def heatmap_generation(
        self, 
        labels: List[int], 
        ascores: Tensor, 
        imgs: Tensor, 
        gtmaps: Tensor = None, 
        grads: Tensor = None, 
        show_per_cls: int = 20,
        name='heatmaps', 
        subdir='.',
        net: FCDDNet = None,
    ):
        minsamples = min(collections.Counter(labels).values())
        lbls = torch.IntTensor(labels)

        if minsamples < 2:
            self.logger.warning(
                f"Heatmap '{name}' cannot be generated. For some labels there are too few samples!", unique=False
            )
        else:
            this_show_per_cls = min(show_per_cls, minsamples)
            if this_show_per_cls % 2 != 0:
                this_show_per_cls -= 1
            # Evaluation Picture with 4 rows. Each row splits into 4 subrows with input-output-heatmap-gtm:
            # (1) 20 first nominal samples (2) 20 first anomalous samples
            # (3) 10 most nominal nominal samples - 10 most anomalous nominal samples
            # (4) 10 most nominal anomalies - 10 most anomalous anomalies
            idx = []
            for l in sorted(set(labels)):
                idx.extend((lbls == l).nonzero().squeeze(-1).tolist()[:this_show_per_cls])
            rascores = net.reduce_ascore(ascores)
            k = max(this_show_per_cls // 2, 1)
            for l in sorted(set(labels)):
                lid = set((lbls == l).nonzero().squeeze(-1).tolist())
                sort = [
                    i for i in np.argsort(rascores.detach().reshape(rascores.size(0), -1).sum(1)).tolist() if i in lid
                ]
                idx.extend([*sort[:k], *sort[-k:]])
            self._create_heatmaps_picture(
                idx, name, imgs.shape, subdir, this_show_per_cls, imgs, ascores, grads, gtmaps, labels
            )

        # Concise paper picture: Samples grow from most nominal to most anomalous (equidistant).
        # 2 versions: with local normalization and semi-global normalization
        res = self.resdown * 2  ## Increase resolution limit because there are only a few heatmaps shown here
        rascores = net.reduce_ascore(ascores)
        inpshp = imgs.shape
        for l in sorted(set(labels)):
            lid = set((torch.from_numpy(np.asarray(labels)) == l).nonzero().squeeze(-1).tolist())
            if len(lid) < 1:
                break
            k = min(show_per_cls // 3, len(lid))
            sort = [
                i for i in np.argsort(rascores.detach().reshape(rascores.size(0), -1).sum(1)).tolist() if i in lid
            ]
            splits = np.array_split(sort, k)
            idx = [s[int(n / (k - 1) * len(s)) if n != len(splits) - 1 else -1] for n, s in enumerate(splits)]
            self.logger.logtxt(
                'Interpretation visualization paper image {} indicies for label {}: {}'
                .format('{}_paper_lbl{}'.format(name, l), l, idx)
            )
            self._create_singlerow_heatmaps_picture(
                idx, name, inpshp, l, subdir, res, imgs, ascores, grads, gtmaps, labels
            )

    def _create_heatmaps_picture(self, idx: List[int], name: str, inpshp: torch.Size, subdir: str,
                                 nrow: int, imgs: Tensor, ascores: Tensor, grads: Tensor, gtmaps: Tensor,
                                 labels: List[int], norm: str = 'global'):
        """
        Creates a picture of inputs, heatmaps (either based on ascores or grads, if grads is not None),
        and ground-truth maps (if not None, otherwise omitted). Each row contains nrow many samples.
        One row contains always only one of {input, heatmaps, ground-truth maps}.
        The order of rows thereby is (1) inputs (2) heatmaps (3) ground-truth maps (4) blank.
        For instance, for 20 samples and nrow=10, the picture would show:
            - 10 inputs
            - 10 corresponding heatmaps
            - 10 corresponding ground-truth maps
            - blank
            - 10 inputs
            - 10 corresponding heatmaps
            - 10 corresponding ground-truth maps
        :param idx: limit the inputs (and corresponding other rows) to these indices.
        :param name: name to be used to store the picture.
        :param inpshp: the input shape (heatmaps will be resized to this).
        :param subdir: some subdirectory to store the data in.
        :param nrow: number of images per row.
        :param imgs: the input images.
        :param ascores: anomaly scores.
        :param grads: gradients.
        :param gtmaps: ground-truth maps.
        :param norm: what type of normalization to apply.
            None: no normalization.
            'local': normalizes each heatmap w.r.t. itself only.
            'global': normalizes each heatmap w.r.t. all heatmaps available (without taking idx into account),
                though it is ensured to consider equally many anomalous and nominal samples (if there are e.g. more
                nominal samples, randomly chosen nominal samples are ignored to match the correct amount).
            'semi-global: normalizes each heatmap w.r.t. all heatmaps chosen in idx.
        """
        number_of_rows = int(np.ceil(len(idx) / nrow))
        rows = []
        for s in range(number_of_rows):
            rows.append(self._image_processing(imgs[idx][s * nrow:s * nrow + nrow], inpshp, maxres=self.resdown, qu=1))
            rows.append(
                self._image_processing(
                    ascores[idx][s * nrow:s * nrow + nrow], inpshp, maxres=self.resdown, qu=self.quantile,
                    colorize=True, ref=balance_labels(ascores, labels, False) if norm == 'global' else ascores[idx],
                    norm=norm.replace('semi_', ''),  # semi case is handled in the line above
                )
            )
            if grads is not None:
                rows.append(
                    self._image_processing(
                        grads[idx][s * nrow:s * nrow + nrow], inpshp, self.blur_heatmaps,
                        self.resdown, qu=self.quantile,
                        colorize=True, ref=balance_labels(grads, labels, False) if norm == 'global' else grads[idx],
                        norm=norm.replace('semi_', ''),  # semi case is handled in the line above
                    )
                )
            if gtmaps is not None:
                rows.append(
                    self._image_processing(
                        gtmaps[idx][s * nrow:s * nrow + nrow], inpshp, maxres=self.resdown, norm=None
                    )
                )
            rows.append(torch.zeros_like(rows[-1]))
        name = '{}_{}'.format(name, norm)
        self.logger.imsave(name, torch.cat(rows), nrow=nrow, scale_mode='none', subdir=subdir)

    def _create_singlerow_heatmaps_picture(self, idx: List[int], name: str, inpshp: torch.Size, lbl: int, subdir: str,
                                           res: int, imgs: Tensor, ascores: Tensor, grads: Tensor, gtmaps: Tensor,
                                           labels: List[int]):
        """
        Creates a picture of inputs, heatmaps (either based on ascores or grads, if grads is not None),
        and ground-truth maps (if not None, otherwise omitted).
        Row-wise: (1) inputs (2) heatmaps (3) ground-truth maps.
        Creates one version with local normalization and one with semi_global normalization.
        :param idx: limit the inputs (and corresponding other rows) to these indices.
        :param name: name to be used to store the picture.
        :param inpshp: the input shape (heatmaps will be resized to this).
        :param lbl: label of samples (indices), only used for naming.
        :param subdir: some subdirectory to store the data in.
        :param res: maximum allowed resolution in pixels (images are downsampled if they exceed this threshold).
        :param imgs: the input images.
        :param ascores: anomaly scores.
        :param grads: gradients.
        :param gtmaps: ground-truth maps.
        """
        for norm in ['local', 'global']:
            rows = [self._image_processing(imgs[idx], inpshp, maxres=res, qu=1)]
            rows.append(
                self._image_processing(
                    ascores[idx], inpshp, maxres=res, colorize=True,
                    ref=balance_labels(ascores, labels, False) if norm == 'global' else None,
                    norm=norm.replace('semi_', ''),  # semi case is handled in the line above
                )
            )
            if grads is not None:
                rows.append(
                    self._image_processing(
                        grads[idx], inpshp, self.blur_heatmaps, res, colorize=True,
                        ref=balance_labels(grads, labels, False) if norm == 'global' else None,
                        norm=norm.replace('semi_', ''),  # semi case is handled in the line above
                    )
                )
            if gtmaps is not None:
                rows.append(self._image_processing(gtmaps[idx], inpshp, maxres=res, norm=None))
            tim = torch.cat(rows)
            imname = '{}_paper_{}_lbl{}'.format(name, norm, lbl)
            self.logger.single_save(imname, torch.stack(rows), subdir=pt.join('tims', subdir))
            self.logger.imsave(imname, tim, nrow=len(idx), scale_mode='none', subdir=subdir)

    def _image_processing(self, imgs: Tensor, input_shape: torch.Size, blur: bool = False, maxres: int = 64,
                          qu: float = None, norm: str = 'local', colorize: bool = False, ref: Tensor = None,
                          cmap: str = 'jet') -> Tensor:
        """
        Applies basic image processing techniques, including resizing, blurring, colorizing, and normalizing.
        The resize operation resizes the images automatically to match the input_shape. Other transformations
        are optional. Can be used to create pseudocolored heatmaps!
        :param imgs: a tensor of some images.
        :param input_shape: the shape of the inputs images the data loader returns.
        :param blur: whether to blur the image (has no effect for FCDD anomaly scores, where the
            anomaly scores are upsampled using a Gaussian kernel anyway).
        :param maxres: maximum allowed resolution in pixels (images are downsampled if they exceed this threshold).
        :param norm: what type of normalization to apply.
            None: no normalization.
            'local': normalizes each image w.r.t. itself only.
            'global': normalizes each image w.r.t. to ref (ref defaults to imgs).
        :param qu: quantile used for normalization, qu=1 yields the typical 0-1 normalization.
        :param colorize: whether to colorize grayscaled images using colormaps (-> pseudocolored heatmaps!).
        :param ref: a tensor of images used for global normalization (defaults to imgs).
        :param cmap: the colormap that is used to colorize grayscaled images.
        :return: transformed tensor of images
        """
        imgs = imgs.detach().clone()
        assert imgs.dim() == len(input_shape) == 4  # n x c x h x w
        std = self.gauss_std
        if qu is None:
            qu = self.quantile

        # upsample if necessary (img.shape != input_shape)
        if imgs.shape[2:] != input_shape[2:]:
            assert isinstance(self.net, ReceptiveNet),                 'Some images are not of full resolution, and network is not a receptive net. This should not occur! '
            imgs = self.net.receptive_upsample(imgs, reception=True, std=std)

        # blur if requested
        if blur:
            assert isinstance(self.net, ReceptiveNet)
            r = self.net.reception['r']
            r = (r - 1) if r % 2 == 0 else r
            std = std or kernel_size_to_std(r)
            imgs = gaussian_blur2d(imgs, (r,) * 2, (std,) * 2)

        # downsample if resolution exceeds the limit given with maxres
        if maxres < max(imgs.shape[2:]):
            assert imgs.shape[-2] == imgs.shape[-1], 'Image provided is no square!'
            imgs = F.interpolate(imgs, (maxres, maxres), mode='nearest')

        # apply requested normalization
        if norm is not None:
            apply_norm = {
                'local': self.__local_norm, 'global': self.__global_norm
            }
            imgs = apply_norm[norm](imgs, qu, ref)

        # if image is grayscaled, colorize, i.e. provide a pseudocolored heatmap!
        if colorize:
            imgs = imgs.mean(1).unsqueeze(1)
            imgs = colorize_img([imgs, ], norm=False, cmap=cmap)[0]
        else:
            imgs = imgs.repeat(1, 3, 1, 1) if imgs.size(1) == 1 else imgs

        return imgs

    @staticmethod
    def __global_norm(imgs: Tensor, qu: int, ref: Tensor = None) -> Tensor:
        """
        Applies a global normalization of tensor, s.t. the highest value of the complete tensor is 1 and
        the lowest value is >= zero. Uses a non-linear normalization based on quantiles as explained in the appendix
        of the paper.
        :param imgs: images tensor
        :param qu: quantile used
        :param ref: if this is None, normalizes w.r.t. to imgs, otherwise normalizes w.r.t. to ref.
        """
        ref = ref if ref is not None else imgs
        imgs.sub_(ref.min())
        ref = ref.sub(ref.min())
        quantile = ref.reshape(-1).kthvalue(int(qu * ref.reshape(-1).size(0)))[0]  # qu% are below that
        imgs.div_(quantile)  # (1 - qu)% values will end up being out of scale ( > 1)
        plosses = imgs.clamp(0, 1)  # clamp those
        return plosses

    @staticmethod
    def __local_norm(imgs: Tensor, qu: int, ref: Tensor = None) -> Tensor:
        """
        Applies a local normalization of tensor, s.t. the highest value of each element (dim=0) in the tensor is 1 and
        the lowest value is >= zero. Uses a non-linear normalization based on quantiles as explained in the appendix
        of the paper.
        :param imgs: images tensor
        :param qu: quantile used
        """
        imgs.sub_(imgs.reshape(imgs.size(0), -1).min(1)[0][(...,) + (None,) * (imgs.dim() - 1)])
        quantile = imgs.reshape(imgs.size(0), -1).kthvalue(
            int(qu * imgs.reshape(imgs.size(0), -1).size(1)), dim=1
        )[0]  # qu% are below that
        imgs.div_(quantile[(...,) + (None,) * (imgs.dim() - 1)])
        imgs = imgs.clamp(0, 1)  # clamp those
        return imgs


# In[21]:
# # eval functions

# In[]:

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
def compute_gtmap_roc(
    anomaly_scores,
    original_gtmaps,
    logger: Logger, 
    net: FCDDNet, 
    subdir='.',
):
    """the scores are upsampled to the images' original size and then the ROC is computed."""
    
    # GTMAPS pixel-wise anomaly detection = explanation performance
    logger.print('Computing ROC score')
    
    # Reduces the anomaly score to be a score per pixel (explanation)
    anomaly_scores = anomaly_scores.mean(1).unsqueeze(1)
    
    if isinstance(net, ReceptiveNet):  # Receptive field upsampling for FCDD nets
        anomaly_scores = net.receptive_upsample(anomaly_scores, std=net.gauss_std)
        
    # Further upsampling for original dataset size
    anomaly_scores = torch.nn.functional.interpolate(anomaly_scores, (original_gtmaps.shape[-2:]))
    flat_gtmaps, flat_ascores = original_gtmaps.reshape(-1).int().tolist(), anomaly_scores.reshape(-1).tolist()
    
    fpr, tpr, ths = roc_curve(
        y_true=flat_gtmaps, 
        y_score=flat_ascores,
        drop_intermediate=True,
    )
    
    # reduce the number of points of the curve
    npoints = ths.shape[0]
    
    if npoints > ROC_PR_CURVES_LIMIT_NUMBER_OF_POINTS:
        
        _, fpr = _reduce_curve_number_of_points(
            x=ths, 
            y=fpr, 
            npoints=ROC_PR_CURVES_INTERPOLATION_NUMBER_OF_POINTS,
        )
        ths, tpr = _reduce_curve_number_of_points(
            x=ths, 
            y=tpr, 
            npoints=ROC_PR_CURVES_INTERPOLATION_NUMBER_OF_POINTS,
        )
    
    auc_score = auc(fpr, tpr)
    
    logger.logtxt(f'##### GTMAP ROC TEST SCORE {auc_score} #####', print=True)
    logger.single_plot(
        'gtmap_roc_curve', 
        values=tpr, 
        xs=fpr, 
        xlabel='false positive rate (fpr)', 
        ylabel='true positive rate (tpr)',
        legend=[f'auc={auc_score}'], 
        subdir=subdir
    )
    gtmap_roc_res = {'tpr': tpr, 'fpr': fpr, 'ths': ths, 'auc': auc_score}
    logger.single_save('gtmap_roc', gtmap_roc_res, subdir=subdir,)

    return gtmap_roc_res

@torch.no_grad()
def compute_gtmap_pr(
    anomaly_scores,
    original_gtmaps,
    logger: Logger, 
    net: FCDDNet, 
    subdir='.',
):
    """
    The scores are upsampled to the images' original size and then the PR is computed.
    The scores are normalized between 0 and 1, and interpreted as anomaly "probability".
    """
    
    # GTMAPS pixel-wise anomaly detection = explanation performance
    logger.print('Computing PR score')
    
    # Reduces the anomaly score to be a score per pixel (explanation)
    anomaly_scores = anomaly_scores.mean(1).unsqueeze(1)
    
    if isinstance(net, ReceptiveNet):  # Receptive field upsampling for FCDD nets
        anomaly_scores = net.receptive_upsample(anomaly_scores, std=net.gauss_std)
        
    # Further upsampling for original dataset size
    anomaly_scores = torch.nn.functional.interpolate(anomaly_scores, (original_gtmaps.shape[-2:]))
    flat_gtmaps, flat_ascores = original_gtmaps.reshape(-1).int().tolist(), anomaly_scores.reshape(-1).tolist()
    
    # min_ascore = np.min(flat_ascores)
    # max_ascore = np.max(flat_ascores)
    # if min_ascore == max_ascore:
    #     logger.print('WARNING: min and max anomaly scores are equal, cannot compute PR curve')
    #     return dict()
    # probas = (flat_ascores - min_ascore) / (max_ascore - min_ascore)
    
    # ths = thresholds
    precision, recall, ths = precision_recall_curve(
        y_true=flat_gtmaps, 
        # probas_pred=probas,
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
    
    if npoints > ROC_PR_CURVES_LIMIT_NUMBER_OF_POINTS:
        
        _, precision = _reduce_curve_number_of_points(
            x=ths, 
            y=precision, 
            npoints=ROC_PR_CURVES_INTERPOLATION_NUMBER_OF_POINTS,
        )
        ths, recall = _reduce_curve_number_of_points(
            x=ths, 
            y=recall, 
            npoints=ROC_PR_CURVES_INTERPOLATION_NUMBER_OF_POINTS,
        )
    
    ap_score = average_precision_score(y_true=flat_gtmaps, y_score=flat_ascores)
    
    logger.logtxt(f'##### GTMAP AP TEST SCORE {ap_score} #####', print=True)
    logger.single_plot(
        'gtmap_pr_curve', 
        values=precision, 
        xs=recall, 
        xlabel='recall', 
        ylabel='precision',
        legend=[f'ap={ap_score}'], 
        subdir=subdir,
    )
    gtmap_pr_res = dict(recall=recall, precision=precision, ths=ths, ap=ap_score)
    logger.single_save('gtmap_pr', gtmap_pr_res, subdir=subdir,)
    return gtmap_pr_res

# In[]:

# # training

# In[]:

# the names come from trainer.test()
RunResults = namedtuple('RunResults', ["gtmap_roc", "gtmap_pr",])


from contextlib import contextmanager


class NoWandb:
    
    def __init__(self, *args, **kwargs) -> None:
        pass
    
    def log(self, *args, **kwargs) -> None:
        pass
    

@contextmanager
def no_wandb_init(*args, **kwargs):
    try:
        yield NoWandb()
    finally:
        pass


def run_one(it, **kwargs):
    """
    kwargs should contain all parameters of the setup function in training.setup
    """
    logdir = kwargs["logdir"]
    
    wandb_project = kwargs.pop("wandb_project", None)
    wandb_tags = kwargs.pop("wandb_tags", None) or []
    wandb_profile = kwargs.pop("wandb_profile", False)

    wandb_init = wandb.init if wandb_project is not None else no_wandb_init
            
    with wandb_init(
        name=f"{logdir.parent.parent.name}.{logdir.parent.name}.{logdir.name}",
        project=wandb_project, 
        entity="mines-paristech-cmm",
        config={**kwargs, **dict(it=it)},
        tags=wandb_tags,
    ) as run:
        
        # these pops must particularly come here
        # after the wandb.init() so they are logged    
        kwargs["logdir"] = str(logdir.absolute())
        kwargs["datadir"] = str(Path(kwargs["datadir"]).absolute())
        readme = kwargs.pop("readme")
        kwargs['config'] = f'{json.dumps(kwargs)}\n\n{readme}'

        acc_batches = kwargs.pop('acc_batches', 1)
        epochs = kwargs.pop('epochs')
        load_snapshot = kwargs.pop('load', None)  # pre-trained model, path to model snapshot
        test = kwargs.pop("test")
        normal_class_label = kwargs.pop("normal_class_label")
        
        del kwargs["log_start_time_str"]

        # this was the part
        # setup = trainer_setup(**kwargs)
        # trainer = SuperTrainer(**setup)
        setup: TrainSetup = trainer_setup(**kwargs)
        
        if load_snapshot is None:
            epoch_start = 0
        
        else:
            # trainer.load(load_snapshot)
            # e.g. the gauss_std must be put in the class object
            raise NotImplemented("the kwargs need to be managed in case of loading...")
            epoch_start, kwargs = train_setup_load_snapshot(
                path=load_snapshot,
                net=setup.net,
                opt=setup.opt,
                sched=setup.sched,
                device=setup.device,
            )
            
        trainer = FCDDTrainer(
            net=setup.net,
            logger=setup.logger,
            gauss_std=setup.gauss_std,
            quantile=setup.quantile,
            resdown=setup.resdown,
            blur_heatmaps=setup.blur_heatmaps,
            device=setup.device,
            pixel_level_loss=setup.pixel_level_loss,
        )

        try:
            # this was the part
            # trainer.train(epochs, load, acc_batches=acc_batches)
            # epochs: from kwargs, ok
            # load: from kwargs, ok
            # acc_batches: from kwargs, ok
            trainer.train(
                net=setup.net,
                opt=setup.opt,
                sched=setup.sched,
                logger=setup.logger,
                train_loader=setup.train_loader,
                device=setup.device,
                epochs=epochs - epoch_start, 
                acc_batches=acc_batches,
                wandb=run, 
            )

            if not (test and (epochs > 0 or load_snapshot is not None)):
                return RunResults(gtmap_roc=dict(), gtmap_pr=dict())
                
            labels, loss, anomaly_scores, imgs, outputs, gtmaps = trainer.test(
                net=setup.net, 
                data_loader=setup.test_loader, 
                logger=setup.logger,
                device=setup.device,
            )
            
            trainer.heatmap_generation(
                labels=labels,
                ascores=anomaly_scores,
                imgs=imgs,
                gtmaps=gtmaps,
                name='test_heatmaps',
                net=setup.net,
            )
            
            original_gtmaps = setup.test_loader.dataset.dataset.get_original_gtmaps_normal_class()
            
            rr = RunResults(
                gtmap_roc=compute_gtmap_roc(
                    anomaly_scores=anomaly_scores,
                    original_gtmaps=original_gtmaps,
                    logger=setup.logger,
                    net=setup.net, 
                ),
                gtmap_pr=compute_gtmap_pr(
                    anomaly_scores=anomaly_scores,
                    original_gtmaps=original_gtmaps,
                    logger=setup.logger,
                    net=setup.net, 
                ),
            )
            
            # ========================== WANDB TEST LOG ==========================
            # ========================== WANDB TEST LOG ==========================
            # ========================== WANDB TEST LOG ==========================
            run.log(dict(
                test_rocauc=rr.gtmap_roc["auc"],
                # ========================== ROC CURVE ==========================
                # copied from wandb.plot.roc_curve()
                # debug=wandb.plot.roc_curve(),
                test_roc_curve=wandb.plot_table(
                    vega_spec_name="wandb/area-under-curve/v0",
                    data_table=wandb.Table(
                        columns=["class", "fpr", "tpr"], 
                        data=[
                            [normal_class_label, fpr_, tpr_] 
                            for fpr_, tpr_ in zip(
                                rr.gtmap_roc["fpr"], 
                                rr.gtmap_roc["tpr"],
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
                test_pr_curve=wandb.plot_table(
                    vega_spec_name="wandb/area-under-curve/v0",
                    data_table=wandb.Table(
                        columns=["class", "recall", "precision"], 
                        data=[
                            [normal_class_label, rec_, prec_] 
                            for rec_, prec_ in zip(
                                rr.gtmap_pr["recall"], 
                                rr.gtmap_pr["precision"],
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
            ))
            return rr

        except:
            setup.logger.printlog += traceback.format_exc()
            epochs = "epochs could not be properly saved because the training broke (exception)"
            raise  # the re-raise is executed after the 'finally' clause

        finally:
            # joao: the original code had this comment about logger.print_logs()
            # no finally statement, because that breaks debugger
            # joao: i'm ignoring it to see what happens
            # and it was in the except clause of the BaseRunner.run_one()
            setup.logger.log_prints() 
            
            setup.logger.save()
            setup.logger.plot()
            
            # setup.logger.snapshot(trainer.net, trainer.opt, trainer.sched, epochs)
            train_setup_save_snapshot(
                outfile=str(Path(setup.logger.dir) / f"snapshot.pt"), 
                net=setup.net, 
                opt=setup.opt, 
                sched=setup.sched, 
                epoch=epochs,
                # kwargs 
                gauss_std=setup.gauss_std,
                quantile=setup.quantile,
                resdown=setup.resdown,
                blur_heatmaps=setup.blur_heatmaps,
                pixel_level_loss=setup.pixel_level_loss,
            )
            
                    
def run(**kwargs) -> dict:
    
    original_logdir = kwargs['logdir']
    dataset = kwargs['dataset']
    
    cls_restrictions = kwargs.pop("cls_restrictions", None)
    classes = cls_restrictions or range(dataset_nclasses(dataset))

    number_it = kwargs.pop('it')
    its_restrictions = kwargs.pop("its_restrictions", None)
    its = its_restrictions or range(number_it)

    results = []
    
    for c in classes:
        cls_logdir = original_logdir / f'normal_{c}'
        
        kwargs['normal_class'] = c
        kwargs['normal_class_label'] = dataset_class_labels(dataset)[c]
    
        for i in its:
            it_logdir = cls_logdir / 'it_{}'.format(i)
            res = run_one(it=i, **{**kwargs, **dict(logdir=it_logdir)})  # overwrite logdir
            results.append(dict(class_idx=c, it=i, results=res))

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


# In[ ]:


# this was in run_seeds
# for key in results:
#     plot_many_roc(
#         logdir.replace('{t}', kwargs["log_start_time_str"], results[key],
#         labels=its, mean=True, name=key
#     )
    
# return {key: mean_roc(val) for key, val in results.items()}


# In[ ]:


# this was in run_classes

        # this was in the finally of the class loop
            # print('Plotting ROC for completed classes up to {}...'.format(c))
            # for key in results:
            #     plot_many_roc(
            #         logdir.replace('{t}', kwargs["log_start_time_str"], results[key],
            #         labels=str_labels(kwargs['dataset']), mean=True, name=key
            #     )
                
    # for key in results:
    #     plot_many_roc(
    #         logdir.replace('{t}', kwargs["log_start_time_str"], results[key],
    #         labels=str_labels(kwargs['dataset']), mean=True, name=key
    #     )

