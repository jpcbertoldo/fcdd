#!/usr/bin/env python
# coding: utf-8


import functools
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from fcdd.models.bases import ReceptiveNet
from pytorch_lightning import LightningModule
from pytorch_lightning.trainer.states import RunningStage
from torch import Tensor
from torch.hub import load_state_dict_from_url

from callbacks_dev01 import merge_steps_outputs

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
    
    MODEL_DIR = Path(__file__).parent.parent.parent / 'data' / 'models'
    
    def __init__(
        self, 
        # model
        in_shape: Tuple[int, int, int], 
        gauss_std: float,
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
        
        # for some reason pyttorch lightning needs this specific call super().__init__()
        super().__init__()
        
        self.last_epoch_outputs = None
        
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
        assert x.shape[-2:] == self.in_shape, f"{x.shape[1:]} != {self.in_shape}"
        x = self.features(x)
        x = self.conv_final(x)
        return x
    
    def loss(self, inputs: Tensor, gtmaps: Tensor) -> Tensor:
        """ computes the FCDD """
        
        anomaly_score_maps = self(inputs) 
        
        loss_maps = anomaly_score_maps = anomaly_score_maps ** 2
        anomaly_score_maps = anomaly_score_maps.sqrt()
        
        loss_maps = (loss_maps + 1).sqrt() - 1
        
        gauss_std = self.hparams["gauss_std"]
        anomaly_score_maps = self._receptive_field_net.receptive_upsample(anomaly_score_maps, reception=True, std=gauss_std, cpu=False)
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
    
    def _common_step(self, batch, batch_idx, stage):
        
        inputs, labels, gtmaps = batch
        score_maps, loss_maps, loss = self.loss(inputs=inputs, gtmaps=gtmaps)
        
        # separate normal/anomaly
        with torch.no_grad():
            
            loss_normals = loss_maps[gtmaps == 0]
            loss_normals_mean = loss_normals.mean()
            
            loss_anomalous = loss_maps[gtmaps == 1]
            loss_anomalous_mean = loss_anomalous.mean()
            
            score_mean = score_maps.mean()
            
            score_normals = score_maps[gtmaps == 0]
            score_normals_mean = score_normals.mean()
            
            score_anomalous = score_maps[gtmaps == 1]
            score_anomalous_mean = score_anomalous.mean()

        self.log(f"{stage}/score-mean", score_mean, on_step=False, on_epoch=True)
        self.log(f"{stage}/score-normals-mean", score_normals_mean, on_step=False, on_epoch=True)
        self.log(f"{stage}/score-anomalous-mean", score_anomalous_mean, on_step=False, on_epoch=True)
        
        # the loss name doesnt follow the logic of the other metrics
        # because the name 'loss' is required by pytorch lightning
        self.log(f"{stage}/loss", loss, on_step=False, on_epoch=True)
        self.log(f"{stage}/loss-normals-mean", loss_normals_mean, on_step=False, on_epoch=True)
        self.log(f"{stage}/loss-anomalous-mean", loss_anomalous_mean, on_step=False, on_epoch=True)
        
        return dict(
            # inputs
            inputs=inputs, 
            labels=labels, 
            gtmaps=gtmaps,
            # score 
            score_maps=score_maps, 
            score_normals=score_normals,
            score_anomalous=score_anomalous,
            # loss
            loss_maps=loss_maps, 
            loss_normals=loss_normals,
            loss_anomalous=loss_anomalous,
            loss=loss,
        )
        
    def training_step(self, batch, batch_idx):    
        return self._common_step(batch, batch_idx, stage=RunningStage.TRAINING)
    
    def training_epoch_end(self, outputs) -> None:
        self.last_epoch_outputs = merge_steps_outputs(outputs)
        pass
    
    def validation_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, stage=RunningStage.VALIDATING)
        pass
    
    def validation_epoch_end(self, validation_step_outputs):
        self.last_epoch_outputs = merge_steps_outputs(validation_step_outputs)
        pass
        
    def test_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, stage=RunningStage.TESTING)
        pass
    
    def test_epoch_end(self, test_step_outputs):
        self.last_epoch_outputs = merge_steps_outputs(test_step_outputs)
    
    def teardown(self, stage=None):
        self.last_epoch_outputs = None
        # heatmap_generation()
    
    # DEPRECATE DEPRECATE DEPRECATE DEPRECATE DEPRECATE DEPRECATE DEPRECATE DEPRECATE DEPRECATE DEPRECATE DEPRECATE DEPRECATE
    
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
