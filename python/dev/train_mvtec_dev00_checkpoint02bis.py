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

from scipy.interpolate import interp1d
import numpy as np
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from fcdd.models import choices, load_nets
from fcdd.models.bases import BaseNet, FCDDNet, ReceptiveNet
from fcdd.util.logging import Logger
from fcdd.util.logging import colorize as colorize_img
from kornia import gaussian_blur2d
from sklearn.metrics import roc_auc_score, roc_curve, auc
from torch import Tensor
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset

# from fcdd.datasets.noise import kernel_size_to_std
# from fcdd.training import balance_labels
# from fcdd.training.setup import pick_opt_sched

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

LOSS_PIXEL_LEVEL = "pixel-level"
LOSS_PIXEL_LEVEL_BALANCED = "pixel-level-balanced"
LOSS_PIXEL_LEVEL_FOCAL = "pixel-level-focal"
LOSS_PIXEL_LEVEL_FOCAL2 = "pixel-level-focal2"
LOSS_PIXEL_WISE_AVERAGE_DISTANCES = "pixel-wise-average-distances"
LOSS_PIXEL_WISE_AVERAGE_DISTANCE_PER_IMAGE = "pixel-wise-average-distance-per-image"
LOSS_PIXEL_WISE_AVERAGES_PER_IMAGE = "pixel-wise-averages-per-image"
LOSS_PIXEL_WISE_AVERAGES_PER_IMAGE_BALANCED = "pixel-wise-averages-per-image-balanced"

LOSS_PIXEL_LEVEL_BALANCED_POST_HUBER = "pixel-level-balanced-post-hubert"
LOSS_PIXEL_LEVEL_BALANCED_HUBER_FACTOR_AVERAGE = "pixel-level-balanced-hubert-factor-average"


LOSS_MODES = (
    LOSS_PIXEL_LEVEL, 
    LOSS_PIXEL_LEVEL_BALANCED, 
    LOSS_PIXEL_LEVEL_FOCAL, 
    LOSS_PIXEL_LEVEL_FOCAL2, 
    LOSS_PIXEL_WISE_AVERAGE_DISTANCES, 
    LOSS_PIXEL_WISE_AVERAGE_DISTANCE_PER_IMAGE,
    LOSS_PIXEL_WISE_AVERAGES_PER_IMAGE,
    LOSS_PIXEL_WISE_AVERAGES_PER_IMAGE_BALANCED,
    LOSS_PIXEL_LEVEL_BALANCED_POST_HUBER,
    LOSS_PIXEL_LEVEL_BALANCED_HUBER_FACTOR_AVERAGE,
)

# In[]:
# # datasets

from mvtec_dataset import ADMvTec, TorchvisionDataset

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
        '-n', '--net', type=str, default='FCDD_CNN224_VGG_F', choices=choices(),
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
        "--loss-mode", type=str, choices=LOSS_MODES, default=LOSS_PIXEL_LEVEL,
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
        net='FCDD_CNN224_VGG_F', 
        dataset='mvtec', 
        noise_mode='confetti',
        oe_limit=1,
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
        "dataset_loaders",
        "opt",
        "sched",
        "logger",
        "device",
        "quantile",
        "resdown",
        "gauss_std",
        "blur_heatmaps",
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
    
    loaders = ds.loaders(batch_size=batch_size, num_workers=workers)
    
    # ================================ NET ================================
    net = load_nets(name=net, in_shape=ds.shape, bias=bias)
    net = net.to(device)
    
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
        dataset_loaders=loaders, 
        opt=optimizer, 
        sched=scheduler, 
        logger=logger,
        device=device, 
        quantile=quantile, 
        resdown=resdown,
        gauss_std=gauss_std, 
        blur_heatmaps=blur_heatmaps,
    )


# In[]:
# # trainer

# In[13]:

class FCDDTrainer:
    
    def __init__(
        self, 
        net: BaseNet, 
        opt: Optimizer, 
        sched: _LRScheduler, 
        dataset_loaders: Tuple[DataLoader, DataLoader],
        logger: Logger, 
        gauss_std: float, 
        quantile: float, 
        resdown: int, 
        loss_mode: str,
        blur_heatmaps=False,
        device='cuda:0',
        **kwargs
    ):
        """
        Anomaly detection trainer that defines a test phase where scores are computed and heatmaps are generated.
        The train method is modified to be able to handle ground-truth maps.
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
        self.opt = opt
        self.sched = sched
        self.train_loader, self.test_loader = dataset_loaders
        self.logger = logger
        self.device = device
        self.gauss_std = gauss_std
        self.quantile = quantile
        self.resdown = resdown
        self.blur_heatmaps = blur_heatmaps
        self.loss_mode = loss_mode
                
    def load(self, path: str, cpu=False) -> int:
        """ Loads a snapshot of the training state, including network weights """
        if cpu:
            snapshot = torch.load(path, map_location=torch.device('cpu'))
        else:
            snapshot = torch.load(path)
        net_state = snapshot.pop('net', None)
        opt_state = snapshot.pop('opt', None)
        sched_state = snapshot.pop('sched', None)
        epoch = snapshot.pop('epoch', None)
        if net_state is not None and self.net is not None:
            self.net.load_state_dict(net_state)
        if opt_state is not None and self.opt is not None:
            self.opt.load_state_dict(opt_state)
        if sched_state is not None and self.sched is not None:
            self.sched.load_state_dict(sched_state)
        print('Loaded {}{}{} with starting epoch {} for {}'.format(
            'net_state, ' if net_state else '', 'opt_state, ' if opt_state else '',
            'sched_state' if sched_state else '', epoch, str(self.__class__)[8:-2]
        ))
        return epoch

    def anomaly_score(self, loss: Tensor) -> Tensor:
        """ This assumes the loss is already the anomaly score. If this is not the case, reimplement the method! """
        return loss

    def reduce_ascore(self, ascore: Tensor) -> Tensor:
        """ Reduces the anomaly score to be a score per image (detection). """
        return ascore.reshape(ascore.size(0), -1).mean(1)

    def reduce_pixelwise_ascore(self, ascore: Tensor) -> Tensor:
        """ Reduces the anomaly score to be a score per pixel (explanation). """
        return ascore.mean(1).unsqueeze(1)

    def train(self, epochs: int, acc_batches=1, wandb = None) -> BaseNet:
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
        
        self.net = self.net.to(self.device).train()
        
        for epoch in range(epochs):
            
            acc_data, acc_counter = [], 1
            
            for n_batch, data in enumerate(self.train_loader):
                
                if acc_counter < acc_batches and n_batch < len(self.train_loader) - 1:
                    acc_data.append(data)
                    acc_counter += 1
                    continue
                elif acc_batches > 1:
                    acc_data.append(data)
                    data = [torch.cat(d) for d in zip(*acc_data)]
                    acc_data, acc_counter = [], 1

                inputs, labels, gtmaps = data
                inputs = inputs.to(self.device)
                gtmaps = gtmaps.to(self.device)
                self.opt.zero_grad()
                outputs = self.net(inputs)
                
                if self.loss_mode == LOSS_PIXEL_WISE_AVERAGE_DISTANCES:
                    loss_norm, loss_anom = self.loss(outputs, inputs, labels, gtmaps)
                    loss_mean = (loss_norm + loss_anom) / 2
                
                else:
                    loss = self.loss(outputs, inputs, labels, gtmaps,)
                    loss_mean = loss.mean()
                
                loss_mean.backward()
                self.opt.step()
                
                with torch.no_grad():
                    self.logger.log(epoch, n_batch, len(self.train_loader), loss_mean,)
                              
            self.net.eval()
            if wandb is not None:

                with torch.no_grad():
                                                                
                    test_score = self.test(subdir=f'test.epoch={epoch:03d}', train_data=False, lite_gtmap=True)
                    
                    wandb.log(dict(
                        epoch=epoch,
                        loss=loss_mean.data.item(),
                        test_rocauc=test_score['gtmap_roc']['auc'],
                        lr=self.opt.param_groups[0]["lr"],
                    ))
                    
            self.net.train()        
            self.sched.step()
                

        return self.net

    def test(self, specific_viz_ids: Tuple[List[int], List[int]] = (), train_data=True, subdir='.', lite_gtmap=False) -> dict:
        """
        Does a full iteration of the data loaders, remembers all data (i.e. inputs, labels, outputs, loss),
        and computes scores and heatmaps with it. Scores and heatmaps are computed for both, the training
        and the test data. For each, one heatmap picture is generated that contains (row-wise):
            -   The first 20 nominal samples (label == 0, if nominal_label==1 this shows anomalies instead).
            -   The first 20 anomalous samples (label == 1, if nominal_label==1 this shows nominal samples instead).
                The :func:`reorder` takes care that the first anomalous test samples are not all from the same class.
            -   The 10 most nominal rated samples from the nominal set on the left and
                the 10 most anomalous rated samples from the nominal set on the right.
            -   The 10 most nominal rated samples from the anomalous set on the left and
                the 10 most anomalous  rated samples from the anomalous set on the right.
        Additionally, for the test set only, four heatmap pictures are generated that show six samples with
        increasing anomaly score from left to right. Thereby the leftmost heatmap shows the most nominal rated example
        and the rightmost sample the most anomalous rated one. There are two heatmaps for the anomalous set and
        two heatmaps for the nominal set. Both with either local normalization -- i.e. each heatmap is normalized
        w.r.t itself only, there is a complete red and complete blue pixel in each heatmap -- or semi-global
        normalization -- each heatmap is normalized w.r.t. to all heatmaps shown in the picture.
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
        self.net = self.net.to(self.device).eval()

        if train_data:
            self.logger.print('Test training data...', fps=False)
            labels, loss, anomaly_scores, imgs, outputs, gtmaps, grads = self._gather_data(
                self.train_loader
            )
            self.heatmap_generation(labels, anomaly_scores, imgs, gtmaps, grads, name='train_heatmaps',)
            
        else:
            self.logger.print('Test training data SKIPPED', fps=False)

        self.logger.print('Test test data...', fps=False)
        labels, loss, anomaly_scores, imgs, outputs, gtmaps, grads = self._gather_data(
            self.test_loader,
        )
        
        def reorder(labels: List[int], loss: Tensor, anomaly_scores: Tensor, imgs: Tensor, outputs: Tensor, gtmaps: Tensor,
                    grads: Tensor, ds: Dataset = None) -> Tuple[List[int], Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
            """ returns all inputs in an identical new order if the dataset offers a predefined (random) order """
            if ds is not None and hasattr(ds, 'fixed_random_order'):
                assert gtmaps is None,                     'original gtmaps loaded in score do not know order! Hence reordering is not allowed for GT datasets'
                o = ds.fixed_random_order
                labels = labels[o] if isinstance(labels, (Tensor, np.ndarray)) else np.asarray(labels)[o].tolist()
                loss, anomaly_scores, imgs = loss[o], anomaly_scores[o], imgs[o]
                outputs, gtmaps = outputs[o], gtmaps
                grads = grads[o] if grads is not None else None
            return labels, loss, anomaly_scores, imgs, outputs, gtmaps, grads
        
        labels, loss, anomaly_scores, imgs, outputs, gtmaps, grads = reorder(
            labels, loss, anomaly_scores, imgs, outputs, gtmaps, grads, ds=self.test_loader.dataset
        )
        self.heatmap_generation(labels, anomaly_scores, imgs, gtmaps, grads, name='test_heatmaps',)

        with torch.no_grad():
            sc = self.score(labels, anomaly_scores, imgs, outputs, gtmaps, grads, subdir=subdir, lite_gtmap=lite_gtmap)
        return sc

    def _gather_data(self, loader: DataLoader,
                     gather_all=False) -> Tuple[List[int], Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        all_labels, all_loss, all_anomaly_scores, all_imgs, all_outputs = [], [], [], [], []
        all_gtmaps, all_grads = [], []
        for n_batch, data in enumerate(loader):
            inputs, labels, gtmaps = data
            all_gtmaps.append(gtmaps)
            bk_inputs = inputs.detach().clone()
            inputs = inputs.to(self.device)
            if gather_all:
                outputs, loss, anomaly_score, _ = self._regular_forward(inputs, labels)
                inputs = bk_inputs.clone().to(self.device)
                _, _, _, grads = self._grad_forward(inputs, labels)
            else:
                outputs, loss, anomaly_score, grads = self._regular_forward(inputs, labels)
            all_labels += labels.detach().cpu().tolist()
            all_loss.append(loss.detach().cpu())
            all_anomaly_scores.append(anomaly_score.detach().cpu())
            all_imgs.append(inputs.detach().cpu())
            all_outputs.append(outputs.detach().cpu())
            if grads is not None:
                all_grads.append(grads.detach().cpu())
            self.logger.print(
                'TEST {:04d}/{:04d} ID {}{}'.format(
                    n_batch, len(loader), str(self.__class__)[8:-2],
                    ' NCLS {}'.format(loader.dataset.normal_classes)
                    if hasattr(loader.dataset, 'normal_classes') else ''
                ),
                fps=True
            )
        all_imgs = torch.cat(all_imgs)
        all_outputs = torch.cat(all_outputs)
        all_gtmaps = torch.cat(all_gtmaps) if len(all_gtmaps) > 0 else None
        all_loss = torch.cat(all_loss)
        all_anomaly_scores = torch.cat(all_anomaly_scores)
        all_grads = torch.cat(all_grads) if len(all_grads) > 0 else None
        ret = (
            all_labels, all_loss, all_anomaly_scores, all_imgs, all_outputs, all_gtmaps,
            all_grads
        )
        return ret

    def _regular_forward(self, inputs: Tensor, labels: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        with torch.no_grad():
            outputs = self.net(inputs)
            loss = self.loss(outputs, inputs, labels)
            anomaly_score = self.anomaly_score(loss)
            grads = None
        return outputs, loss, anomaly_score, grads

    def _grad_forward(self, inputs: Tensor, labels: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        inputs.requires_grad = True
        outputs = self.net(inputs)
        loss = self.loss(outputs, inputs, labels)
        anomaly_score = self.anomaly_score(loss)
        grads = self.net.get_grad_heatmap(loss, inputs)
        inputs.requires_grad = False
        self.opt.zero_grad()
        return outputs, loss, anomaly_score, grads

    def score(self, labels: List[int], ascores: Tensor, imgs: Tensor, outs: Tensor, gtmaps: Tensor = None,
              grads: Tensor = None, subdir='.', lite_gtmap=False) -> dict:
        """
        Computes the ROC curves and the AUC for detection performance.
        Also computes those for the explanation performance if ground-truth maps are available.
        :param labels: labels
        :param ascores: anomaly scores
        :param imgs: input images
        :param outs: outputs of the neural network
        :param gtmaps: ground-truth maps (can be None)
        :param grads: gradients of anomaly scores w.r.t. inputs (can be None)
        :param subdir: subdirectory to store the data in (plots and numbers)
        :return:  A dictionary of ROC results, each ROC result is again represented by a dictionary of the form: {
                'tpr': [], 'fpr': [], 'ths': [], 'auc': int, ...
            }.
        """
        # Logging
        self.logger.print('Computing test score...')
        if torch.isnan(ascores).sum() > 0:
            self.logger.logtxt('Could not compute test scores, since anomaly scores contain nan values!!!', True)
            return None
        red_ascores = self.reduce_ascore(ascores).tolist()
        std = self.gauss_std

        # Overall ROC for sample-wise anomaly detection
        fpr, tpr, thresholds = roc_curve(labels, red_ascores)
        roc_score = roc_auc_score(labels, red_ascores)
        roc_res = {'tpr': tpr, 'fpr': fpr, 'ths': thresholds, 'auc': roc_score}
        self.logger.single_plot(
            'roc_curve', tpr, fpr, xlabel='false positive rate', ylabel='true positive rate',
            legend=['auc={}'.format(roc_score)], subdir=subdir
        )
        self.logger.single_save('roc', roc_res, subdir=subdir)
        self.logger.logtxt('##### ROC TEST SCORE {} #####'.format(roc_score), print=True)

        # GTMAPS pixel-wise anomaly detection = explanation performance
        gtmap_roc_res, gtmap_prc_res = None, None
        use_grads = grads is not None
        if gtmaps is not None:
            try:
                self.logger.print('Computing GT test score...')
                ascores = self.reduce_pixelwise_ascore(ascores) if not use_grads else grads
                gtmaps = self.test_loader.dataset.dataset.get_original_gtmaps_normal_class()
                if isinstance(self.net, ReceptiveNet):  # Receptive field upsampling for FCDD nets
                    ascores = self.net.receptive_upsample(ascores, std=std)
                # Further upsampling for original dataset size
                ascores = torch.nn.functional.interpolate(ascores, (gtmaps.shape[-2:]))
                flat_gtmaps, flat_ascores = gtmaps.reshape(-1), ascores.reshape(-1)
                
                if lite_gtmap:
                    import random
                    indices = torch.tensor(random.sample(range(len(flat_gtmaps)), 3000), device=self.device)
                    flat_gtmaps = flat_gtmaps[indices]
                    flat_ascores = flat_ascores[indices]
                    
                flat_gtmaps, flat_ascores = flat_gtmaps.int().tolist(), flat_ascores.tolist()
                
                gtfpr, gttpr, gtthresholds = roc_curve(
                    y_true=flat_gtmaps, 
                    y_score=flat_ascores,
                    drop_intermediate=True,
                )
                
                # reduce the number of points of the ROC curve
                ROC_LIMIT_NUMBER_OF_POINTS = 6000
                ROC_INTERPOLATION_NUMBER_OF_POINTS = 3000
                roc_npoints = gtthresholds.shape[0]
                if roc_npoints > ROC_LIMIT_NUMBER_OF_POINTS:
                    
                    func_fpr = interp1d(gtthresholds, gtfpr, kind='linear')
                    func_tpr = interp1d(gtthresholds, gttpr, kind='linear')

                    thmin, thmax = np.min(gtthresholds), np.max(gtthresholds)
                    gtthresholds = np.linspace(thmin, thmax, ROC_INTERPOLATION_NUMBER_OF_POINTS, endpoint=True)
                    
                    gtfpr = func_fpr(gtthresholds)
                    gttpr = func_tpr(gtthresholds)
                
                gt_roc_score = auc(gtfpr, gttpr)
                gtmap_roc_res = {'tpr': gttpr, 'fpr': gtfpr, 'ths': gtthresholds, 'auc': gt_roc_score}
                
                self.logger.single_plot(
                    'gtmap_roc_curve', 
                    gttpr, 
                    gtfpr, 
                    xlabel='false positive rate', 
                    ylabel='true positive rate',
                    legend=['auc={}'.format(gt_roc_score)], 
                    subdir=subdir
                )
                self.logger.single_save(
                    'gtmap_roc', 
                    gtmap_roc_res, 
                    subdir=subdir,
                )
                self.logger.logtxt('##### GTMAP ROC TEST SCORE {} #####'.format(gt_roc_score), print=True)
            except AssertionError as e:
                self.logger.warning(f'Skipped computing the gtmap ROC score. {str(e)}')

        return {'roc': roc_res, 'gtmap_roc': gtmap_roc_res}

    def heatmap_generation(
        self, 
        labels: List[int], 
        ascores: Tensor, 
        imgs: Tensor, 
        gtmaps: Tensor = None, 
        grads: Tensor = None, 
        show_per_cls: int = 20,
        name='heatmaps', 
        subdir='.'
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
            rascores = self.reduce_ascore(ascores)
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
        if 'train' not in name:
            res = self.resdown * 2  ## Increase resolution limit because there are only a few heatmaps shown here
            rascores = self.reduce_ascore(ascores)
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

    def loss(self, outs: Tensor, ins: Tensor, labels: Tensor, gtmaps: Tensor = None):
        """ computes the FCDD """
        
        assert isinstance(self.net, FCDDNet)
        
        # outs \in R^{N x d x H x W}
        loss = (outs ** 2).sum(dim=1, keepdim=True)
        loss = (loss + 1).sqrt() - 1
        
        if not self.net.training:
            return loss  # here it is always loss map
        
        assert gtmaps is not None
        
        std = self.gauss_std
        loss = self.net.receptive_upsample(loss, reception=True, std=std, cpu=False)
         
        if self.loss_mode == LOSS_PIXEL_LEVEL_BALANCED_POST_HUBER:
            loss = (outs ** 2).sum(dim=1, keepdim=True).sqrt()
            
            loss = self.net.receptive_upsample(loss, reception=True, std=self.gauss_std, cpu=False)
            
            # these have raw/converted values or 0, and they are complementary
            norm_loss_maps = (loss * (1 - gtmaps))
            anom_loss_maps = -(((1 - (-loss).exp()) + 1e-31).log())
            anom_loss_maps = anom_loss_maps * gtmaps

            # apply the hubert trick
            loss = (loss**2 + 1).sqrt() - 1

            # balancing ratio
            n_pixels_normal = (1 - gtmaps).sum()
            n_pixels_anomalous = gtmaps.sum()
            ratio_norm_anom = n_pixels_normal / (n_pixels_anomalous + 1)
            
            # combine them such that there is no 0, so the batch-wise average can be computed
            loss = norm_loss_maps + ratio_norm_anom * anom_loss_maps   
            
            batch_size = loss.size(0)
            return loss.view(batch_size, -1).mean(-1)
            
        if self.loss_mode == LOSS_PIXEL_LEVEL_BALANCED_HUBER_FACTOR_AVERAGE:
            
            loss = (outs ** 2).sum(dim=1, keepdim=True)
            
            # apply the hubert trick with a factor computed from the average distance
            avg_dist = loss.sqrt().mean()
            loss = (((loss / (avg_dist  + 1e-31)) + 1).sqrt() - 1) * avg_dist.sqrt()
            
            loss = self.net.receptive_upsample(loss, reception=True, std=self.gauss_std, cpu=False)
            
            # these have raw/converted values or 0, and they are complementary
            norm_loss_maps = (loss * (1 - gtmaps))
            anom_loss_maps = -(((1 - (-loss).exp()) + 1e-31).log())
            anom_loss_maps = anom_loss_maps * gtmaps

            # balancing ratio
            n_pixels_normal = (1 - gtmaps).sum()
            n_pixels_anomalous = gtmaps.sum()
            ratio_norm_anom = n_pixels_normal / (n_pixels_anomalous + 1)
            
            # combine them such that there is no 0, so the batch-wise average can be computed
            loss = norm_loss_maps + ratio_norm_anom * anom_loss_maps   
            
            batch_size = loss.size(0)
            return loss.view(batch_size, -1).mean(-1)
        
        # ============================================================
        
        if self.loss_mode == LOSS_PIXEL_WISE_AVERAGE_DISTANCES:
            # here i'm actually taking the average distances in the batch
            # so the two sums (on i: image, and j: pixel of the image) go inside the log
            norm_avg_loss = (loss[(1 - gtmaps).bool()]).mean()  # for normals dist = loss
            anom_avg_dist = (loss[gtmaps.bool()]).mean()
            anom_avg_loss = -(((1 - (-anom_avg_dist).exp()) + 1e-31).log())
            return norm_avg_loss, anom_avg_loss
        
        # this is the one described in paper
        if self.loss_mode == LOSS_PIXEL_WISE_AVERAGE_DISTANCE_PER_IMAGE:
            # we want to take the average distance of only the normal pixels in each image 
            # (then the same with the anomalous pixels)
            # the code below is strange but it works  
            # if you are not convinced, then try to uncomment the following lines and check that it works
            #
            # import numpy as np
            # import torch as tc
            # x = tc.stack([i * tc.tensor(range(1, 5)).reshape(2, 2) for i in range(1, 4)]).double()
            # print("axes: [page, row column]")
            # print("x")
            # print(x)
            # print("task: on each page, take the average of the odd numbers")
            # print("the odd numbers are ate positions:")
            # print(x % 2 == 1)
            # print("the expected answer is: [2, 0, 6]        obs: the 0 in position [1] is arbitrary (it'll be used for a loss later...)")
            # t = tc.full_like(x, np.nan)
            # print('t')
            # print(t)
            # t[x % 2 == 1] = x[x % 2 == 1]
            # print("t after copying the relevant data from x")
            # print(t)
            # # torch 1.11.0 has Tensor.nanmean() but not here yet...
            # notnan = ~t.isnan()
            # n_notnan = notnan.sum(dim=(-2, -1))
            # print("number of not nan per page")
            # print(n_notnan)
            # ans = t.nansum(dim=(-2, -1)) / (n_notnan + 1e-31)
            # print('ans')
            # print(ans)
            
            # norm_loss \in R^(N, 1, H, W)
            norm_loss = torch.full_like(loss, np.nan)
            norm_loss[(1 - gtmaps).bool()] = loss[(1 - gtmaps).bool()]
            norm_select = ~norm_loss.isnan()
            # norm_loss \in R^(N,)
            norm_n_notnan = norm_select.sum(dim=(-3, -2, -1)) 
            norm_loss = norm_loss.nansum(dim=(-3, -2, -1)) / (norm_n_notnan + 1e-31)  # +1e-31 to avoid division by 0 (x.nansum() returns 0 if all are nan)
            
            # anom_loss \in R^(N, 1, H, W)
            anom_loss = torch.full_like(loss, np.nan)
            anom_loss[gtmaps.bool()] = loss[gtmaps.bool()]
            anom_select = ~anom_loss.isnan()
            # anom_loss \in R^(N,)
            anom_n_notnan = anom_select.sum(dim=(-3, -2, -1))
            anom_loss = anom_loss.nansum(dim=(-3, -2, -1)) / (anom_n_notnan + 1e-31)  # +1e-31 to avoid division by 0 (x.nansum() returns 0 if all are nan)
            
            # now both losses are in R^(N,) so loss \in R^(N,)
            # at this point, each position in norm_loss[i] and anom_loss[i] is the average distance 
            # of the normal pixels and the anomalous pixels in the i-th image respectively
            loss = norm_loss - ((1 - (-anom_loss).exp()) + 1e-31).log()
            return loss
        
        if self.loss_mode == LOSS_PIXEL_WISE_AVERAGES_PER_IMAGE or self.loss_mode == LOSS_PIXEL_WISE_AVERAGES_PER_IMAGE_BALANCED:
            # in this one i do the average per image as above
            # but i do the conversion of anomalous loss before the per-image averaging
            
            # norm_loss \in R^(N, 1, H, W)
            norm_loss = torch.full_like(loss, np.nan)
            norm_loss[(1 - gtmaps).bool()] = loss[(1 - gtmaps).bool()]
            norm_select = ~norm_loss.isnan()
            # norm_loss \in R^(N,)
            norm_n_notnan = norm_select.sum(dim=(-3, -2, -1)) 
            norm_loss = norm_loss.nansum(dim=(-3, -2, -1)) / (norm_n_notnan + 1e-31)  # +1e-31 to avoid division by 0 (x.nansum() returns 0 if all are nan)
            
            # anom_loss \in R^(N, 1, H, W)
            anom_loss = torch.full_like(loss, np.nan)
            # anom_loss[gtmaps.bool()] = 
            anom_loss[gtmaps.bool()] = - ((1 - (-loss[gtmaps.bool()]).exp()) + 1e-31).log()
            anom_select = ~anom_loss.isnan()
            # anom_loss \in R^(N,)
            anom_n_notnan = anom_select.sum(dim=(-3, -2, -1))
            anom_loss = anom_loss.nansum(dim=(-3, -2, -1)) / (anom_n_notnan + 1e-31)  # +1e-31 to avoid division by 0 (x.nansum() returns 0 if all are nan)
            
            if self.loss_mode == LOSS_PIXEL_WISE_AVERAGES_PER_IMAGE_BALANCED:
                # apply a balancing factor to the anomalous to compensate the fact that 
                # usually there are less anomalous pixels
                assert len(gtmaps.shape) == 4, f"gtmaps.shape = {gtmaps.shape}"
                n_pixels_norm_perimg = (1 - gtmaps).sum(dim=(-3, -2, -1))  # \in R^(N,)
                n_pixels_anom_perimg = gtmaps.sum(dim=(-3, -2, -1))  # \in R^(N,)
                assert len(n_pixels_norm_perimg.shape) == 1, f"n_pixels_norm_perimg.shape = {n_pixels_norm_perimg.shape}"
                ratio_norm_anom_perimg = n_pixels_norm_perimg / (n_pixels_anom_perimg + 1)
                anom_loss = ratio_norm_anom_perimg * anom_loss
                
            # now both losses are in R^(N,) so loss \in R^(N,)
            # at this point, each position in norm_loss[i] and anom_loss[i] is the average distance 
            # of the normal pixels and the anomalous pixels in the i-th image respectively
            loss = norm_loss + anom_loss
            return loss
            
        
        # ============================= pixel wise losses =============================
        # for the losses here below we keep the whole tensor structure all through the computation
        # then at the end we take the mean over each image
        
        norm_loss_maps = (loss * (1 - gtmaps))
        anom_loss_maps = -(((1 - (-loss).exp()) + 1e-31).log())
        anom_loss_maps = anom_loss_maps * gtmaps

        if self.loss_mode == LOSS_PIXEL_LEVEL:
            loss = norm_loss_maps + anom_loss_maps                    
        
        elif self.loss_mode == LOSS_PIXEL_LEVEL_BALANCED:
            n_pixels_normal = (1 - gtmaps.cpu()).sum()
            n_pixels_anomalous = gtmaps.cpu().sum()
            ratio_norm_anom = n_pixels_normal / (n_pixels_anomalous + 1)
            loss = norm_loss_maps + ratio_norm_anom * anom_loss_maps                    
        
        elif self.loss_mode == LOSS_PIXEL_LEVEL_FOCAL:
            loss = norm_loss_maps + anom_loss_maps
            loss_np = loss.cpu().detach().numpy()                    
            import random
            from scipy.stats import percentileofscore
            loss_sample = loss_np.ravel()[random.sample(range(loss_np.size), 100)]
            def percent_score_func(value):
                return percentileofscore(loss_sample, value)
            percent_score_func = np.vectorize(percent_score_func)
            weights = percent_score_func(loss_np)
            weights = torch.tensor(weights / weights.sum(), device=self.device)
            loss = loss * weights                            
        
        elif self.loss_mode == LOSS_PIXEL_LEVEL_FOCAL2:

            n_pixels_normal = int((1 - gtmaps).sum())
            n_pixels_anomalous = int(gtmaps.sum())
            
            loss = norm_loss_maps + anom_loss_maps
            loss_np = loss.cpu().detach().numpy()  
            gtmaps_np = gtmaps.cpu().detach().numpy()
                                
            import random
            from scipy.stats import percentileofscore
            
            normal_loss_sample = (loss_np[(1 - gtmaps_np).astype(bool)]).ravel()[random.sample(range(n_pixels_normal), 500)]
            anomalous_loss_sample = (loss_np[gtmaps_np.astype(bool)]).ravel()[random.sample(range(n_pixels_anomalous), 500)]
            loss_sample = np.concatenate((normal_loss_sample, anomalous_loss_sample))
            
            def percent_score_func(value):
                return percentileofscore(loss_sample, value)
            percent_score_func = np.vectorize(percent_score_func)
            
            weights = percent_score_func(loss_np)
            weights = torch.tensor(weights / weights.sum(), device=self.device)
            
            loss = loss * weights                                   
            
        else:
            raise NotImplementedError(f'Loss mode {self.loss_mode} not implemented!')
                    
        batch_size = loss.size(0)
        return loss.view(batch_size, -1).mean(-1)
        

# In[15]:

def use_wandb() -> bool:
    return bool(int(os.environ.get("WANDB", "0")))

# In[21]:

# the names come from trainer.test()
RunResults = namedtuple('RunResults', ["roc", "gtmap_roc",])


def run_one(it, **kwargs):
    """
    kwargs should contain all parameters of the setup function in training.setup
    """
    logdir = kwargs["logdir"]
    
    if use_wandb():
        import wandb
        wandb.init(
            name=f"{logdir.parent.parent.name}.{logdir.parent.name}.{logdir.name}",
            project="fcdd-mvtec-dev00-checkpoint02", 
            entity="mines-paristech-cmm",
            config={**kwargs, **dict(it=it)},
            tags=["fix-test-no-grad", "checkpoint02bis"],
            settings=wandb.Settings(start_method="fork"),
        )
        
    kwargs["logdir"] = str(logdir.absolute())
    kwargs["datadir"] = str(Path(kwargs["datadir"]).absolute())
    readme = kwargs.pop("readme")
    kwargs['config'] = f'{json.dumps(kwargs)}\n\n{readme}'

    acc_batches = kwargs.pop('acc_batches', 1)
    epochs = kwargs.pop('epochs')
    load_snapshot = kwargs.pop('load', None)  # pre-trained model, path to model snapshot
    test = kwargs.pop("test")
    loss_mode = kwargs.pop("loss_mode")
    
    del kwargs["log_start_time_str"]
    del kwargs["normal_class_label"]
    
    try:
        # this was the part
        # setup = trainer_setup(**kwargs)
        # trainer = SuperTrainer(**setup)
        setup: TrainSetup = trainer_setup(**kwargs)
        trainer = FCDDTrainer(
            net=setup.net,
            opt=setup.opt,
            sched=setup.sched,
            dataset_loaders=setup.dataset_loaders,
            logger=setup.logger,
            gauss_std=setup.gauss_std,
            quantile=setup.quantile,
            resdown=setup.resdown,
            blur_heatmaps=setup.blur_heatmaps,
            device=setup.device,
            loss_mode=loss_mode,
        )
        
        if load_snapshot is None:
            epoch_start = 0
        
        else:
            epoch_start = trainer.load(load_snapshot)
            
    except:
        if use_wandb():
            import wandb
            wandb.finish()
        raise

    try:
        # this was the part
        # trainer.train(epochs, load, acc_batches=acc_batches)
        # epochs: from kwargs, ok
        # load: from kwargs, ok
        # acc_batches: from kwargs, ok
        trainer.train(
            epochs=epochs - epoch_start, 
            acc_batches=acc_batches,
            wandb=wandb if use_wandb() else None, 
        )

        if test and (epochs > 0 or load_snapshot is not None):
            ret = trainer.test()  # keys = {roc, gtmap_roc}
            rr = RunResults(
                roc=ret["roc"],
                gtmap_roc=ret["gtmap_roc"],
            )
            
            if use_wandb():
                wandb.log(dict(test_rocauc=ret["gtmap_roc"]["auc"]))
            
            return rr
        else:
            return RunResults({}, {})
        
    except:
        setup.logger.printlog += traceback.format_exc()
        raise  # the re-raise is executed after the 'finally' clause

    finally:
        # joao: the original code had this comment about logger.print_logs()
        # no finally statement, because that breaks debugger
        # joao: i'm ignoring it to see what happens
        # and it was in the except clause of the BaseRunner.run_one()
        setup.logger.log_prints() 
        
        setup.logger.save()
        setup.logger.plot()
        setup.logger.snapshot(trainer.net, trainer.opt, trainer.sched, epochs)

        if use_wandb():
            wandb.finish()    
            
                    
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

