#!/usr/bin/env python
# coding: utf-8


import functools
import pytorch_lightning as pl


import contextlib
import os.path as pt
import pathlib
import random
import time
from argparse import ArgumentParser
from collections import namedtuple
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from fcdd.models.bases import ReceptiveNet
from pytorch_lightning import LightningModule
from pytorch_lightning.loggers import WandbLogger
from torch import Tensor
from torch.hub import load_state_dict_from_url
from torch.profiler import tensorboard_trace_handler

import mvtec_dataset_dev01 as mvtec_dataset_dev01
import wandb
from common_dev01 import create_seed, seed_int2str, seed_str2int
from data_dev01 import (ANOMALY_TARGET, NOMINAL_TARGET,
                        generate_dataloader_images,)


OPTIMIZER_SGD = 'sgd'
OPTIMIZER_ADAM = 'adam'
OPTIMIZER_CHOICES = (OPTIMIZER_SGD, OPTIMIZER_ADAM)
print(f"OPTIMIZER_CHOICES={OPTIMIZER_CHOICES}")

SCHEDULER_LAMBDA = 'lambda'
SCHEDULER_CHOICES = (SCHEDULER_LAMBDA,)
print(f"SCHEDULER_CHOICES={SCHEDULER_CHOICES}")

LOSS_PIXELWISE_BATCH_AVG = 'pixelwise_batch_avg'
LOSS_CHOICES = (LOSS_PIXELWISE_BATCH_AVG,)


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
        pixel_level_loss: bool,
        # optimizer
        optimizer_name: str,
        lr: float,
        weight_decay: float,
        # scheduler
        scheduler_name: str,
        scheduler_paramaters: list,
        # else
        loss_name: str,
    ):
        assert optimizer_name in OPTIMIZER_CHOICES
        assert scheduler_name in SCHEDULER_CHOICES
        assert loss_name in LOSS_CHOICES
        
        super().__init__()
        
        self._receptive_field_net = ReceptiveNet((3,) + in_shape, bias=True)
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
        
        if optimizer_name == OPTIMIZER_SGD:
            return optim.SGD(
                self.parameters(), 
                lr=lr, 
                weight_decay=weight_decay, 
                momentum=0.9, 
                nesterov=True
            )
        
        elif optimizer_name == OPTIMIZER_ADAM:
            optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        
        else:
            raise NotImplementedError('Optimizer type {} not known.'.format(optimizer_name))
        
        # ================================ scheduler ================================
        
        scheduler_name = self.hparams['scheduler_name']
        lr_sched_param = self.hparams['lr_sched_param']
        
        if scheduler_name == SCHEDULER_LAMBDA:
            
            assert len(lr_sched_param) == 1, 'lambda scheduler needs one parameter' 
            assert 0 < lr_sched_param[0] <= 1, 'lambda scheduler parameter [0] must be in (0, 1]'
            
            scheduler = optim.lr_scheduler.LambdaLR(
                optimizer, 
                lambda ep: lr_sched_param[0] ** ep
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
            
        self.log("train/loss/normal", loss_normal, on_step=False, on_epoch=True)
        self.log("train/loss/anomaly", loss_anomaly, on_step=False, on_epoch=True)
        self.log("train/loss", loss, on_step=False, on_epoch=True)
        
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
                        ["anomalous", fpr_, tpr_] 
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
                        ["anomalous", rec_, prec_] 
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
