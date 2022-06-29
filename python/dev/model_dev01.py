#!/usr/bin/env python
# coding: utf-8


from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
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
SCHEDULER_CHOICES = (None, SCHEDULER_LAMBDA,)
print(f"SCHEDULER_CHOICES={SCHEDULER_CHOICES}")

LOSS_OLD_FCDD = 'old-fcdd'
LOSS_PIXELWISE_BATCH_AVG = 'pixelwise-batch-avg'
LOSS_CHOICES = (LOSS_PIXELWISE_BATCH_AVG, LOSS_OLD_FCDD)


class OptimizersMixin:
    
    def configure_optimizer_sgd(self, lr: float, weight_decay: float) -> optim.Optimizer:
        assert weight_decay >= 0, f"weight_decay={weight_decay}"
        assert lr > 0, f"lr={lr}"
        return optim.SGD(
            self.parameters(), 
            lr=lr, 
            weight_decay=weight_decay, 
            momentum=0.9, 
            nesterov=True,
        )
    
    def configure_optimizer_adam(self, lr: float, weight_decay: float) -> optim.Optimizer:
        assert weight_decay >= 0, f"weight_decay={weight_decay}"
        assert lr > 0, f"lr={lr}"
        return optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
    
    def configure_scheduler_lambda(self, optimizer, lr_exp_decrease_rate):
        assert 0 < lr_exp_decrease_rate < 1, f"lr_exp_decrease_rate={lr_exp_decrease_rate}"
        return optim.lr_scheduler.LambdaLR(optimizer, lambda ep: lr_exp_decrease_rate ** ep)    


class PixelwiseHSCLossesMixin:
    """HSC stands for HyperSphere Classifier"""
    
    def hsc_loss(
        self, 
        scores: Tensor, 
        masks: Tensor, 
        labels: Tensor, 
        loss_version: str, 
        extra_return: Optional[dict] = None,
    ) -> Tuple[Tensor, Tensor]:
        
        assert loss_version in LOSS_CHOICES, f"loss_version={loss_version} not in {LOSS_CHOICES}"
        
        assert scores.shape[1:] == masks.shape[1:], f"{scores.shape[1:]} != {masks.shape[1:]}"
        assert scores.shape[0] == masks.shape[0] == labels.shape[0], f"{scores.shape[0]} != {masks.shape[0]} != {labels.shape[0]}"
        
        assert (scores >= 0).all(), f"{scores.min()} < 0"
        
        unique_labels = tuple(sorted(labels.unique().tolist()))
        assert unique_labels in ((0,), (1,), (0, 1)), f"labels not \in {{0, 1}}, unique_labels={unique_labels}"
        
        unique_mask_values = tuple(sorted(masks.unique().tolist()))
        assert unique_mask_values in ((0,), (1,), (0, 1)), f"mask values not \in {{0, 1}}, unique_mask_values={unique_mask_values}"
        
        if extra_return is not None:
            assert isinstance(extra_return, dict), f"extra_return={extra_return} not dict"
            assert len(extra_return) == 0, f"extra_return={extra_return} not empty"
        
        if loss_version == LOSS_OLD_FCDD:
            loss_maps = self._old_fcdd(scores, masks)
        
        elif loss_version == LOSS_PIXELWISE_BATCH_AVG:
            loss_maps = self._pixel_wise_batch_avg(scores, masks, extra_return)
            
        else:
            raise NotImplementedError(f"loss '{loss_version}' not implemented")
        
        return loss_maps, loss_maps.mean()

    def _pixel_wise_batch_avg(self, scores, masks, extra_return):
        
        # scores \in R+^{N x 1 x H x W}
        loss_maps = (scores + 1).sqrt() - 1
            
        # normal term is kept the same
        norm_loss_maps = (loss_maps * (1 - masks))
            
            # anomalous term is pushed
        anom_loss_maps = - (((1 - (-loss_maps).exp()) + 1e-31).log())
        anom_loss_maps = anom_loss_maps * masks
            
        n_pixels_normal = (1 - masks).sum()
        n_pixels_anomalous = masks.sum()
        ratio_norm_anom = n_pixels_normal / (n_pixels_anomalous + 1)
                      
        loss_maps = norm_loss_maps + ratio_norm_anom * anom_loss_maps     
            
        if extra_return is not None:
            extra_return['ratio_norm_anom'] = ratio_norm_anom
            extra_return["loss_maps_nobalance"] = norm_loss_maps + anom_loss_maps
            
        return loss_maps

    def _old_fcdd(self, scores, masks):
        # scores \in R+^{N x 1 x H x W}
        loss_maps = (scores + 1).sqrt() - 1
            
        norm_loss_maps = (loss_maps * (1 - masks))
            
        anom_loss_maps = -(((1 - (-loss_maps).exp()) + 1e-31).log())
        anom_loss_maps = anom_loss_maps * masks
            
        loss_maps = norm_loss_maps + anom_loss_maps
        
        return loss_maps


class FCDD_CNN224_VGG_F(OptimizersMixin, PixelwiseHSCLossesMixin, LightningModule):
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
        self.in_shape = in_shape
        
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
            torch.nn.Conv2d(3, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            torch.nn.MaxPool2d(2, 2),
            torch.nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            torch.nn.MaxPool2d(2, 2),
            torch.nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            torch.nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            torch.nn.MaxPool2d(2, 2),
            # Frozen version freezes up to here
            torch.nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            torch.nn.Conv2d(512, 512, 3, 1, 1),
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
        
        self.conv_final = torch.nn.Conv2d(512, 1, 1)
        
        self.save_hyperparameters()
        pass
    
    def forward(self, x):
        assert x.shape[-2:] == self.in_shape, f"{x.shape[1:]} != {self.in_shape}"
        x = self.features(x)
        x = self.conv_final(x)
        return x
    
    def loss(self, scores: Tensor, masks: Tensor, labels: Tensor) -> Tuple[Tensor, Tensor]:
        pass 
    
    def configure_optimizers(self):
        
        # =================================== optimizer =================================== 
        optimizer_name = self.hparams['optimizer_name']
        assert optimizer_name in OPTIMIZER_CHOICES, f"optimizer '{optimizer_name}' unknown"
        
        lr = self.hparams['lr']
        weight_decay = self.hparams['weight_decay']
        
        if optimizer_name == OPTIMIZER_SGD:
            optimizer = self.configure_optimizer_sgd(lr, weight_decay)
        
        elif optimizer_name == OPTIMIZER_ADAM:
            optimizer = self.configure_optimizer_adam(lr, weight_decay)
        
        else:
            raise NotImplementedError(f'Optimizer type {optimizer_name} not known.')
        
        # ================================ scheduler ================================
        
        scheduler_name = self.hparams['scheduler_name']
        assert scheduler_name in SCHEDULER_CHOICES, f"scheduler '{scheduler_name}' unknown"
        
        if scheduler_name is None:
            return [optimizer]
        
        scheduler_paramaters = self.hparams['scheduler_paramaters']
        
        if scheduler_name == SCHEDULER_LAMBDA:
            assert len(scheduler_paramaters) == 1, 'lambda scheduler needs 1 parameter' 
            scheduler = self.configure_scheduler_lambda(optimizer, lr_exp_decrease_rate=scheduler_paramaters[0])
            
        else:
            raise NotImplementedError(f'LR scheduler type {scheduler_name} not known.')
        
        return [optimizer], [scheduler] 
    
    @property
    def reception(self):
        """
        receptive field specifically hard-coded for the archicteture FCDD_CNN224_VGG
        """
        return {
            'n': 28, 'j': 8, 'r': 62, 's': 3.5, 'img_shape': (3, 224, 224),
            # !!!
            # this one didnt exist before, i hard-coded it in there to further simplify
            # this is only valid for the class FCDD_CNN224_VGG_F
            # !!!
            'std': 12,
        }
    
    def _common_step(self, batch, batch_idx, stage):
        
        # inputs \in [0, 1]^{N x C=3 x H x W}
        # labels \in {0, 1}^{N}
        # gtmaps \in {0, 1}^{N x 1 x H x W}
        inputs, labels, gtmaps = batch
        # score_maps \in {0, 1}^{N x 1 x H x W}
        score_maps = self(inputs) ** 2
        
        def receptive_upsample(pixels: torch.Tensor) -> torch.Tensor:
            """
            Implement this to upsample given tensor images based on the receptive field with a Gaussian kernel.
            """
            assert pixels.dim() == 4 and pixels.size(1) == 1, 'receptive upsample works atm only for one channel'
            
            pixels = pixels.squeeze(1)
            
            ishape = self.reception['img_shape']
            pixshp = pixels.shape
            
            # regarding s: if between pixels, pick the first
            s, j, r = int(self.reception['s']), self.reception['j'], self.reception['r']
            
            # !!!
            # this one didnt exist before, i hard-coded it in there to further simplify
            # this is only valid for the class FCDD_CNN224_VGG_F
            # !!!
            std = self.reception['std']
            
            def gkern(k: int, std: float = None):
                "" "Returns a 2D Gaussian kernel array with given kernel size k and std std """
                from scipy import signal
                if k % 2 == 0:
                    # if kernel size is even, signal.gaussian returns center values sampled from gaussian at x=-1 and x=1
                    # which is much less than 1.0 (depending on std). Instead, sample with kernel size k-1 and duplicate center
                    # value, which is 1.0. Then divide whole signal by 2, because the duplicate results in a too high signal.
                    gkern1d = signal.gaussian(k - 1, std=std).reshape(k - 1, 1)
                    gkern1d = np.insert(gkern1d, (k - 1) // 2, gkern1d[(k - 1) // 2]) / 2
                else:
                    gkern1d = signal.gaussian(k, std=std).reshape(k, 1)
                gkern2d = np.outer(gkern1d, gkern1d)
                return gkern2d
            
            gaus = torch.from_numpy(gkern(r, std)).float().to(pixels.device)
            pad = (r - 1) // 2
            
            if (r - 1) % 2 == 0:
                res = torch.nn.functional.conv_transpose2d(
                    pixels.unsqueeze(1), gaus.unsqueeze(0).unsqueeze(0), stride=j, padding=0,
                    output_padding=ishape[-1] - (pixshp[-1] - 1) * j - 1
                )
                
            else:
                res = torch.nn.functional.conv_transpose2d(
                    pixels.unsqueeze(1), gaus.unsqueeze(0).unsqueeze(0), stride=j, padding=0,
                    output_padding=ishape[-1] - (pixshp[-1] - 1) * j - 1 - 1
                )
            out = res[:, :, pad - s:-pad - s, pad - s:-pad - s]  # shift by receptive center (s)
            return out
            
        score_maps = receptive_upsample(score_maps)
        loss_maps, loss = self.hsc_loss(scores=score_maps, masks=gtmaps, labels=labels, loss_version=self.hparams["loss_name"])
        
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
    
    def validation_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, stage=RunningStage.VALIDATING)
    
    def validation_epoch_end(self, validation_step_outputs):
        self.last_epoch_outputs = merge_steps_outputs(validation_step_outputs)
        
    def test_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, stage=RunningStage.TESTING)
    
    def test_epoch_end(self, test_step_outputs):
        self.last_epoch_outputs = merge_steps_outputs(test_step_outputs)
    
    def teardown(self, stage=None):
        self.last_epoch_outputs = None
    