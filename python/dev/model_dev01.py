#!/usr/bin/env python
# coding: utf-8


from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from pytorch_lightning import LightningModule
from pytorch_lightning.trainer.states import RunningStage
from torch import Tensor
from torch.hub import load_state_dict_from_url
import torch
from torch import Tensor
from typing import Tuple, Optional
import numpy as np


from callbacks_dev01 import merge_steps_outputs
from common_dev01 import AdaptiveClipError, find_scores_clip_values_from_empircal_cdf

OPTIMIZER_SGD = 'sgd'
OPTIMIZER_ADAM = 'adam'
OPTIMIZER_CHOICES = (OPTIMIZER_SGD, OPTIMIZER_ADAM)
print(f"OPTIMIZER_CHOICES={OPTIMIZER_CHOICES}")

SCHEDULER_LAMBDA = 'lambda'
SCHEDULER_CHOICES = (None, SCHEDULER_LAMBDA,)
print(f"SCHEDULER_CHOICES={SCHEDULER_CHOICES}")

# fcdd losses, based on a single score map output
LOSS_OLD_FCDD = 'old-fcdd'
LOSS_PIXELWISE_BATCH_AVG = 'pixelwise-batch-avg'
LOSS_PIXELWISE_BATCH_AVG_CLIP_SCORE_CDF_ADAPTIVE = 'pixelwise-batch-avg-clip-score-cdf-adaptive'
LOSS_FCDD_CHOICES = (LOSS_OLD_FCDD, LOSS_PIXELWISE_BATCH_AVG, LOSS_PIXELWISE_BATCH_AVG_CLIP_SCORE_CDF_ADAPTIVE)

# **u2net losses**
# un2net outputs as many score maps as there are heights + 1, the fused score map
# so we cand apply the loss to all sides (as in the original paper)
# or only to the fuse
# in any case we still reuse the same losses defined for

LOSS_U2NET_SCOREMAP_STRATEGY_ALLSIDES = 'allsides'
LOSS_U2NET_SCOREMAP_STRATEGY_FUSEDONLY = 'fusedonly'
LOSS_U2NET_SCOREMAP_STRATEGY_CHOICES = (
    LOSS_U2NET_SCOREMAP_STRATEGY_ALLSIDES, LOSS_U2NET_SCOREMAP_STRATEGY_FUSEDONLY,
)

LOSS_U2NET_ALLSIDES_OLD_FCDD = 'u2net-allsides-old-fcdd'
LOSS_U2NET_ALLSIDES_PIXELWISE_BATCH_AVG = 'u2net-allsides-pixelwise-batch-avg'
LOSS_U2NET_ALLSIDES_CHOICES = (
    LOSS_U2NET_ALLSIDES_OLD_FCDD, LOSS_U2NET_ALLSIDES_PIXELWISE_BATCH_AVG,
)

LOSS_U2NET_FUSEDONLY_OLD_FCDD = 'u2net-fusedonly-old-fcdd'
LOSS_U2NET_FUSEDONLY_PIXELWISE_BATCH_AVG = 'u2net-fusedonly-pixelwise-batch-avg'
LOSS_U2NET_FUSEDONLY_CHOICES = (
    LOSS_U2NET_FUSEDONLY_OLD_FCDD, LOSS_U2NET_FUSEDONLY_PIXELWISE_BATCH_AVG,
)

LOSS_U2NET_CHOICES = LOSS_U2NET_ALLSIDES_CHOICES + LOSS_U2NET_FUSEDONLY_CHOICES

def loss_u2net_parse(loss_name: str) -> Tuple[str, str]:
    """
    un2net outputs as many score maps as there are heights + 1, the fused score map
    the u2net losses contain two information: 
        (1) fcdd loss (loss applied to each score map) 
        (2) score map strategy (which score map(s) it should be applied to)
    this function will parse and return (1) and (2) in this order
    """
    if loss_name == LOSS_U2NET_ALLSIDES_OLD_FCDD:
        return LOSS_OLD_FCDD, LOSS_U2NET_SCOREMAP_STRATEGY_ALLSIDES
    
    elif loss_name == LOSS_U2NET_ALLSIDES_PIXELWISE_BATCH_AVG:
        return LOSS_PIXELWISE_BATCH_AVG, LOSS_U2NET_SCOREMAP_STRATEGY_ALLSIDES
    
    elif loss_name == LOSS_U2NET_FUSEDONLY_OLD_FCDD:
        return LOSS_OLD_FCDD, LOSS_U2NET_SCOREMAP_STRATEGY_FUSEDONLY
    
    elif loss_name == LOSS_U2NET_FUSEDONLY_PIXELWISE_BATCH_AVG:
        return LOSS_PIXELWISE_BATCH_AVG, LOSS_U2NET_SCOREMAP_STRATEGY_FUSEDONLY
    
    else:
        raise NotImplementedError(f"loss_name={loss_name}")


LOSS_CHOICES = LOSS_FCDD_CHOICES + LOSS_U2NET_CHOICES


class SchedulersMixin:
    
    def configure_scheduler_lambda(self, optimizer, lr_exp_decrease_rate):
        assert 0 < lr_exp_decrease_rate < 1, f"lr_exp_decrease_rate={lr_exp_decrease_rate}"
        return optim.lr_scheduler.LambdaLR(optimizer, lambda ep: lr_exp_decrease_rate ** ep)    


class PixelwiseHSCLossesMixin:
    """
    Different versions of the pixel-wise HSC loss. 
    HSC stands for HyperSphere Classifier
    """
    
    def hsc_loss(
        self, 
        score_map: Tensor, 
        masks: Tensor, 
        labels: Tensor, 
        loss_version: str, 
        extra_return: Optional[dict] = None,
    ) -> Tuple[Tensor, Tensor]:
        
        assert loss_version in LOSS_CHOICES, f"loss_version={loss_version} not in {LOSS_CHOICES}"
        
        assert score_map.shape[1:] == masks.shape[1:], f"{score_map.shape[1:]} != {masks.shape[1:]}"
        assert score_map.shape[0] == masks.shape[0] == labels.shape[0], f"{score_map.shape[0]} != {masks.shape[0]} != {labels.shape[0]}"
        
        assert (score_map >= 0).all(), f"{score_map.min()} < 0"
        
        unique_labels = tuple(sorted(labels.unique().tolist()))
        assert unique_labels in ((0,), (1,), (0, 1)), f"labels not \in {{0, 1}}, unique_labels={unique_labels}"
        
        unique_mask_values = tuple(sorted(masks.unique().tolist()))
        assert unique_mask_values in ((0,), (1,), (0, 1)), f"mask values not \in {{0, 1}}, unique_mask_values={unique_mask_values}"
        
        if extra_return is not None:
            assert isinstance(extra_return, dict), f"extra_return={extra_return} not dict"
            assert len(extra_return) == 0, f"extra_return={extra_return} not empty"
        
        if loss_version == LOSS_OLD_FCDD:
            loss_map = self._old_fcdd(score_map, masks)
        
        elif loss_version == LOSS_PIXELWISE_BATCH_AVG:
            loss_map = self._pixel_wise_batch_avg(score_map, masks, extra_return)
        
        elif loss_version == LOSS_PIXELWISE_BATCH_AVG_CLIP_SCORE_CDF_ADAPTIVE:
            loss_map = self._pixel_wise_batch_avg_clip_score_cdf_adaptive(score_map, masks)
            
        else:
            raise NotImplementedError(f"loss '{loss_version}' not implemented")
        
        return loss_map, loss_map.mean()
    
    def _pixel_wise_batch_avg_clip_score_cdf_adaptive(self, score_map, masks):
        
        try:
            with torch.no_grad():
                # temporary hack to not modify the cli
                import wandb
                import os
                loss_empirical_cdf_clip_threshold = float(os.environ.get('loss_empirical_cdf_clip_threshold', 0.05))
                wandb.run.summary.update(dict(loss_empirical_cdf_clip_threshold=loss_empirical_cdf_clip_threshold))
                clipmin, clipmax = find_scores_clip_values_from_empircal_cdf(
                    scores_normal=score_map[masks == 0], 
                    scores_anomalous=score_map[masks == 1],
                    cutfunc_threshold=loss_empirical_cdf_clip_threshold,
                    # cutfunc_threshold=0.05,
                )

        except AdaptiveClipError as ex:
            warnings.warn(f"AdaptiveClipError: clipping could not be applied, using default values: {ex}", stacklevel=2)
        
        else:
            score_map = score_map.clamp(min=clipmin, max=clipmax)
            
        return self._pixel_wise_batch_avg(score_map, masks)

    def _pixel_wise_batch_avg(self, scores, masks, extra_return: Optional[dict] = None):
        
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


MODEL_FCDD_CNN224_VGG_F = "FCDD_CNN224_VGG_F"  # this const is like this for backward compatibility
MODEL_CHOICES_FCDD = (
    MODEL_FCDD_CNN224_VGG_F,
)


class FCDD(SchedulersMixin, PixelwiseHSCLossesMixin, LightningModule):
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
        in_shape: Tuple[int, int], 
        # optimizer
        optimizer_name: str,
        lr: float,
        weight_decay: float,
        model_name: str,
        # scheduler
        scheduler_name: str,
        scheduler_parameters: list,
        # else
        loss_name: str,
    ):
        assert optimizer_name in OPTIMIZER_CHOICES, f"optimizer_name={optimizer_name} not in {OPTIMIZER_CHOICES}"
        assert scheduler_name in SCHEDULER_CHOICES, f"scheduler_name={scheduler_name} not in {SCHEDULER_CHOICES}"
        assert loss_name in LOSS_FCDD_CHOICES, f"loss_name={loss_name} not in {LOSS_FCDD_CHOICES}"
        assert model_name in MODEL_CHOICES_FCDD, f"model_name={model_name} not in {MODEL_CHOICES_FCDD}"
        
        if model_name != MODEL_FCDD_CNN224_VGG_F:
            raise NotImplementedError(f"model_name={model_name} not implemented")
        
        # for some reason pyttorch lightning needs this specific call super().__init__()
        super().__init__()
        
        self.last_epoch_outputs = None
        self.in_shape = tuple(in_shape)
        
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
        assert tuple(x.shape[-2:]) == self.in_shape, f"{x.shape[-2:]} != {self.in_shape}"
        x = self.features(x)
        x = self.conv_final(x)
        return x
    
    def configure_optimizers(self):
        
        # =================================== optimizer =================================== 
        optimizer_name = self.hparams['optimizer_name']
        assert optimizer_name in OPTIMIZER_CHOICES, f"optimizer '{optimizer_name}' unknown"
        
        lr = self.hparams['lr']
        weight_decay = self.hparams['weight_decay']
        
        if optimizer_name == OPTIMIZER_SGD:
            optimizer = optim.SGD(
                self.parameters(), 
                lr=lr, 
                weight_decay=weight_decay, 
                momentum=0.9, 
                nesterov=True,
            )
        
        elif optimizer_name == OPTIMIZER_ADAM:
            optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        
        else:
            raise NotImplementedError(f'Optimizer type {optimizer_name} not known.')
        
        # ================================ scheduler ================================
        
        scheduler_name = self.hparams['scheduler_name']
        assert scheduler_name in SCHEDULER_CHOICES, f"scheduler '{scheduler_name}' unknown"
        
        if scheduler_name is None:
            return [optimizer]
        
        scheduler_parameters = self.hparams['scheduler_parameters']
        
        if scheduler_name == SCHEDULER_LAMBDA:
            assert len(scheduler_parameters) == 1, 'lambda scheduler needs 1 parameter' 
            scheduler = self.configure_scheduler_lambda(optimizer, lr_exp_decrease_rate=scheduler_parameters[0])
            
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
        loss_maps, loss = self.hsc_loss(score_map=score_maps, masks=gtmaps, labels=labels, loss_version=self.hparams["loss_name"])
        
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


from typing import Dict, Tuple
from fcdd.models.bases import FCDDNet
import torch
import torch.nn as nn

import math


def _upsample_like(x, size):
    return nn.Upsample(size=size, mode='bilinear', align_corners=False)(x)


def _stage_sizes_map(in_shape, height):
    """
    in_shape: (W, H)
    
    {height: size} for Upsample
    example
        1: (128, 128)
        2: (64, 64)
        3: (32, 32)
        4: (16, 16)
    """
    sizes = {
        1: tuple(in_shape),
    }
    
    for h in range(2, height + 1):
        previous_size = sizes[h - 1]
        sizes[h] = tuple(math.ceil(w / 2) for w in previous_size)
        
    return sizes


class REBNCONV(nn.Module):
    
    def __init__(self, in_ch=3, out_ch=3, dilate=1):
        super(REBNCONV, self).__init__()

        self.conv_s1 = nn.Conv2d(in_ch, out_ch, 3, padding='same', dilation=1 * dilate)
        self.bn_s1 = nn.BatchNorm2d(out_ch)
        # self.relu_s1 = nn.ReLU(inplace=True)
        # replaced by
        self.relu_s1 = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        return self.relu_s1(self.bn_s1(self.conv_s1(x)))


class RepVGGConv(nn.Module):
    
    def __init__(self, in_ch=3, out_ch=3, dilate=1):
        super(RepVGGConv, self).__init__()

        self.conv_33 = nn.Conv2d(in_ch, out_ch, 3, padding="same", dilation=1 * dilate)
        self.bn_33 = nn.BatchNorm2d(out_ch)
        
        self.conv_11 = nn.Conv2d(in_ch, out_ch, 1, padding="same", dilation=1 * dilate)
        self.bn_11 = nn.BatchNorm2d(out_ch)
        
        self.in_and_out_equal = in_ch == out_ch
        self.bn_id = nn.BatchNorm2d(out_ch) if self.in_and_out_equal else None
        
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        x_33 = self.bn_33(self.conv_33(x))
        x_11 = self.bn_11(self.conv_11(x))
        
        summed = x_33 + x_11
        
        if self.in_and_out_equal:
            summed += self.bn_id(x)
            
        return self.relu(summed)


U2NET_BASIC_BLOCK_REBNCONV = "rebnconv"
U2NET_BASIC_BLOCK_REPVGG = "repvgg"
U2NET_BASIC_BLOCK_CHOICES = (U2NET_BASIC_BLOCK_REBNCONV, U2NET_BASIC_BLOCK_REPVGG,)


class RSU(nn.Module):
    
    def __init__(self, name, height, in_ch, mid_ch, out_ch, dilated=False, basic_block=U2NET_BASIC_BLOCK_REBNCONV):
        
        super(RSU, self).__init__()
        
        self.name = name
        self.height = height
        self.dilated = dilated
        
        assert basic_block in U2NET_BASIC_BLOCK_CHOICES, f"basic_block must be one of {U2NET_BASIC_BLOCK_CHOICES}"
        
        self.basic_block = basic_block
        
        if basic_block == U2NET_BASIC_BLOCK_REBNCONV:
            self.basic_block_class = REBNCONV
        
        elif basic_block == U2NET_BASIC_BLOCK_REPVGG:
            self.basic_block_class = RepVGGConv
        
        else:
            raise ValueError(f"basic_block must be one of {U2NET_BASIC_BLOCK_CHOICES}")
        
        self._make_layers(height, in_ch, mid_ch, out_ch, dilated)

    def forward(self, x):
        sizes = _stage_sizes_map(x.shape[-2:], self.height)
        x = self.rebnconvin(x)

        # U-Net like symmetric encoder-decoder structure
        def unet(x, height=1):
            if height < self.height:
                x1 = getattr(self, f'rebnconv{height}')(x)
                if not self.dilated and height < self.height - 1:
                    x2 = unet(getattr(self, 'downsample')(x1), height + 1)
                else:
                    x2 = unet(x1, height + 1)

                x = getattr(self, f'rebnconv{height}d')(torch.cat((x2, x1), 1))
                return _upsample_like(x, sizes[height - 1]) if not self.dilated and height > 1 else x
            else:
                return getattr(self, f'rebnconv{height}')(x)

        return x + unet(x)

    def _make_layers(self, height, in_ch, mid_ch, out_ch, dilated=False):
        self.add_module('rebnconvin', self.basic_block_class(in_ch, out_ch))
        self.add_module('downsample', nn.MaxPool2d(2, stride=2, ceil_mode=True))

        self.add_module(f'rebnconv1', self.basic_block_class(out_ch, mid_ch))
        self.add_module(f'rebnconv1d', self.basic_block_class(mid_ch * 2, out_ch))

        for i in range(2, height):
            dilate = 1 if not dilated else 2 ** (i - 1)
            self.add_module(f'rebnconv{i}', self.basic_block_class(mid_ch, mid_ch, dilate=dilate))
            self.add_module(f'rebnconv{i}d', self.basic_block_class(mid_ch * 2, mid_ch, dilate=dilate))

        dilate = 2 if not dilated else 2 ** (height - 1)
        self.add_module(f'rebnconv{height}', self.basic_block_class(mid_ch, mid_ch, dilate=dilate))


# LAYER NAMING CONVENTIONS

# def rsu_name2height(rsu_name: str) -> int:
#     h = int(rsu_name.split("_")[-1])
#     assert 0 < h < 10 
#     return h

def rsu_name(height_idx: int, is_decoder: bool) -> str:
    if is_decoder:
        return f"De_{height_idx}"
    else:
        return f"En_{height_idx}"

def side_name(height_: int) -> str:
    return f'side{height_}'


def side_finalconv_name(height_: int) -> str:
    return f'sidefinalconv{height_}'


def downsample_name(height_: int) -> str:
    return f'downsample{height_}'


def upsample_name(height_: int) -> str:
    return f'upsample{height_}'


def score_upsample_name(height_: int) -> str:
    return f'scoreupsample{height_}'



def stage_name(height_idx: int, isdecoder: bool) -> str:
    return f'stage{height_idx}{"d" if isdecoder else ""}'


def parse_stage_name(name: str) -> Tuple[int, bool]:
    """str -> (stage index, is_decoder (bool))"""
    original_name = name
    assert name.startswith("stage"), f"{original_name} is not a stage name"
    name = name.lstrip("stage")
    assert name.endswith("d") or name[-1].isdigit(), f"{original_name} is not a stage name"
    height, isdecoder = (name[:-1], True) if name.endswith("d") else (name, False)
    height = int(height)
    assert 0 < height <= 9, f"{height} is not a valid stage height (i think something may bug if height > 9 because of some stupid naming convention)"
    return height, isdecoder


MODEL_U2NET_HEIGHT6_FULL = "u2net-height6-full"
MODEL_U2NET_HEIGHT6_LITE = "u2net-height6-lite"
MODEL_U2NET_HEIGHT4_LITE = "u2net-height4-lite"
MODEL_U2NET_REPVGG_HEIGHT4_LITE = "u2net-repvgg-height4-lite"
MODEL_U2NET_REPVGG_HEIGHT5_FULL = "u2net-repvgg-height5-full"
MODEL_HYPERSPHERE_U2NET_CHOICES = (
    MODEL_U2NET_HEIGHT6_FULL,
    MODEL_U2NET_HEIGHT6_LITE,
    MODEL_U2NET_HEIGHT4_LITE,
    MODEL_U2NET_REPVGG_HEIGHT4_LITE,
    MODEL_U2NET_REPVGG_HEIGHT5_FULL,
)

HYPERSHPERE_U2NET_LOSS_WEIGHTS_UNIFORM = "uniform"


class HyperSphereU2Net(
    SchedulersMixin, 
    PixelwiseHSCLossesMixin, 
    LightningModule,
):
    """Class for u-net-style networks that are trained to have pixe-wise data descriptors"""
    
    def __init__(
        self, 
        in_shape: Tuple[int, int], 
        optimizer_name: str,
        lr: float,
        weight_decay: float,
        scheduler_name: str,
        scheduler_parameters: list,
        loss_name: str,
        model_name: str,
        # this is held constant for now (there is no option in the cli)
        loss_stage_weights: Union[str, Tuple[float, ...]] = HYPERSHPERE_U2NET_LOSS_WEIGHTS_UNIFORM,
    ):
        assert optimizer_name in OPTIMIZER_CHOICES, f"{optimizer_name} is not a valid optimizer name"
        assert scheduler_name in SCHEDULER_CHOICES, f"{scheduler_name} is not a valid scheduler name"
        assert loss_name in LOSS_U2NET_CHOICES, f"{loss_name} is not a valid loss name"
        assert model_name in MODEL_HYPERSPHERE_U2NET_CHOICES, f"{model_name} not in {MODEL_HYPERSPHERE_U2NET_CHOICES}"
        
        assert len(in_shape) == 2, f"{in_shape} is not a valid input shape"
        assert in_shape[0] == in_shape[1], f"{in_shape} is not a square input shape"
        assert 0 < in_shape[0], f"{in_shape} is not a valid input shape"
        
        # parse the loss name
        loss_name, loss_scoremap_strategy = loss_u2net_parse(loss_name)
        assert loss_name in LOSS_FCDD_CHOICES, f"{loss_name} is not a valid loss name"
        assert loss_scoremap_strategy in LOSS_U2NET_SCOREMAP_STRATEGY_CHOICES, f"{loss_scoremap_strategy} is not a valid loss scoremap strategy"
        
        self.hparams["loss_name"] = loss_name
        self.hparams["loss_scoremap_strategy"] = loss_scoremap_strategy
        
        # for some reason pyttorch lightning needs this specific call super().__init__()
        super().__init__()
        
        self._configure_architecture(model_name, in_shape, loss_stage_weights)

        self.last_epoch_outputs = None
        self.save_hyperparameters()
    
    def _configure_architecture(self, model, in_shape: Tuple[int,], loss_stage_weights):
        
        assert model in MODEL_HYPERSPHERE_U2NET_CHOICES, f"{model} not in {MODEL_HYPERSPHERE_U2NET_CHOICES}"

        # cfgs for building RSUs and sides
        # each stage in the dictionary is (name, RSU_config, side)
        # name: str -> see parse_stage_name() and stage_name()
        # side: int -> nb. of channels for the latent space at that level
        # RSU_config: (height: int, in_ch: int, mid_ch: int, out_ch: int, dilated: bool)
        if model == MODEL_U2NET_HEIGHT6_FULL:
            self.architecture_config = {
                'stage1': ((7, 3, 32, 64, False, U2NET_BASIC_BLOCK_REBNCONV), None),
                'stage2': ((6, 64, 32, 128, False, U2NET_BASIC_BLOCK_REBNCONV), None),
                'stage3': ((5, 128, 64, 256, False, U2NET_BASIC_BLOCK_REBNCONV), None),
                'stage4': ((4, 256, 128, 512, False, U2NET_BASIC_BLOCK_REBNCONV), None),
                'stage5': ((4, 512, 256, 512, True, U2NET_BASIC_BLOCK_REBNCONV), None),
                'stage6': ((4, 512, 256, 512, True, U2NET_BASIC_BLOCK_REBNCONV), 512),
                'stage5d': ((4, 1024, 256, 512, True, U2NET_BASIC_BLOCK_REBNCONV), 512),
                'stage4d': ((4, 1024, 128, 256, False, U2NET_BASIC_BLOCK_REBNCONV), 256),
                'stage3d': ((5, 512, 64, 128, False, U2NET_BASIC_BLOCK_REBNCONV), 128),
                'stage2d': ((6, 256, 32, 64, False, U2NET_BASIC_BLOCK_REBNCONV), 64),
                'stage1d': ((7, 128, 16, 64, False, U2NET_BASIC_BLOCK_REBNCONV), 64),
            }
        
        elif model == MODEL_U2NET_HEIGHT6_LITE:
            self.architecture_config= {
                'stage1': ((7, 3, 16, 64, False, U2NET_BASIC_BLOCK_REBNCONV), None),
                'stage2': ((6, 64, 16, 64, False, U2NET_BASIC_BLOCK_REBNCONV), None),
                'stage3': ((5, 64, 16, 64, False, U2NET_BASIC_BLOCK_REBNCONV), None),
                'stage4': ((4, 64, 16, 64, False, U2NET_BASIC_BLOCK_REBNCONV), None),
                'stage5': ((4, 64, 16, 64, True, U2NET_BASIC_BLOCK_REBNCONV), None),
                'stage6': ((4, 64, 16, 64, True, U2NET_BASIC_BLOCK_REBNCONV), 64),
                'stage5d': ((4, 128, 16, 64, True, U2NET_BASIC_BLOCK_REBNCONV), 64),
                'stage4d': ((4, 128, 16, 64, False, U2NET_BASIC_BLOCK_REBNCONV), 64),
                'stage3d': ((5, 128, 16, 64, False, U2NET_BASIC_BLOCK_REBNCONV), 64),
                'stage2d': ((6, 128, 16, 64, False, U2NET_BASIC_BLOCK_REBNCONV), 64),
                'stage1d': ((7, 128, 16, 64, False, U2NET_BASIC_BLOCK_REBNCONV), 64),
            }

        elif model == MODEL_U2NET_HEIGHT4_LITE:
            self.architecture_config= {
                'stage1': ((7, 3, 16, 64, False, U2NET_BASIC_BLOCK_REBNCONV), None),
                'stage2': ((6, 64, 16, 64, False, U2NET_BASIC_BLOCK_REBNCONV), None),
                'stage3': ((5, 64, 16, 64, False, U2NET_BASIC_BLOCK_REBNCONV), None),
                'stage4': ((4, 64, 16, 64, False, U2NET_BASIC_BLOCK_REBNCONV), 64),
                'stage3d': ((5, 128, 16, 64, False, U2NET_BASIC_BLOCK_REBNCONV), 64),
                'stage2d': ((6, 128, 16, 64, False, U2NET_BASIC_BLOCK_REBNCONV), 64),
                'stage1d': ((7, 128, 16, 64, False, U2NET_BASIC_BLOCK_REBNCONV), 64),
            }
            
        elif model == MODEL_U2NET_REPVGG_HEIGHT4_LITE:
            self.architecture_config= {
                'stage1': ((7, 3, 16, 64, False, U2NET_BASIC_BLOCK_REPVGG), None),
                'stage2': ((6, 64, 16, 64, False, U2NET_BASIC_BLOCK_REPVGG), None),
                'stage3': ((5, 64, 16, 64, False, U2NET_BASIC_BLOCK_REPVGG), None),
                'stage4': ((4, 64, 16, 64, False, U2NET_BASIC_BLOCK_REPVGG), 64),
                'stage3d': ((5, 128, 16, 64, False, U2NET_BASIC_BLOCK_REPVGG), 64),
                'stage2d': ((6, 128, 16, 64, False, U2NET_BASIC_BLOCK_REPVGG), 64),
                'stage1d': ((7, 128, 16, 64, False, U2NET_BASIC_BLOCK_REPVGG), 64),
            }
            
        elif model == MODEL_U2NET_REPVGG_HEIGHT5_FULL:
            self.architecture_config= {
                'stage1': ((7, 3, 32, 64, False, U2NET_BASIC_BLOCK_REPVGG), None),
                'stage2': ((6, 64, 32, 128, False, U2NET_BASIC_BLOCK_REPVGG), None),
                'stage3': ((5, 128, 64, 256, False, U2NET_BASIC_BLOCK_REPVGG), None),
                'stage4': ((4, 256, 128, 512, False, U2NET_BASIC_BLOCK_REPVGG), None),
                'stage5': ((4, 512, 256, 512, True, U2NET_BASIC_BLOCK_REPVGG), 512),
                'stage4d': ((4, 1024, 128, 256, False, U2NET_BASIC_BLOCK_REPVGG), 256),
                'stage3d': ((5, 512, 64, 128, False, U2NET_BASIC_BLOCK_REPVGG), 128),
                'stage2d': ((6, 256, 32, 64, False, U2NET_BASIC_BLOCK_REPVGG), 64),
                'stage1d': ((7, 128, 16, 64, False, U2NET_BASIC_BLOCK_REPVGG), 64),
            }
        
        else:
            raise NotImplementedError(f"{model} not implemented") 
        
        # ============================== parse architecture config ==============================
                
        self.nstages = len(self.architecture_config)
        assert self.nstages % 2 == 1, f"nstages in the config must be odd, but got {self.nstages}"
        
        self.height = int((self.nstages + 1) / 2)
        assert self.height > 1, f"height must be greater than 1, but got {self.height}"
        
        lowest_resolution = in_shape[0] / 2**(self.height - 1)
        assert lowest_resolution > 1, f"lowest resolution must be greater than 1, but got {lowest_resolution}"
        assert lowest_resolution == int(lowest_resolution), f"lowest resolution must be an integer, but got {lowest_resolution}"
        
        for stage_name in self.architecture_config:
            height_idx, isdecoder = parse_stage_name(stage_name)  # just to make sure there is no error
            
            assert 1 <= height_idx <= self.height, f"height_idx must be between 1 and self.height={self.height}, but got height_idx={height_idx}"
        
        # [0] is the stage height idx, [1] is the stage isdecoder
        ndifferent_height_idx = len({parse_stage_name(stage_name)[0] for stage_name in self.architecture_config})
        assert ndifferent_height_idx == self.height, f"there must be exactly {self.height} different height_idx in the config, but got {ndifferent_height_idx}"
                
        # [1] is `isdecoder`
        ndecoders = sum(int(parse_stage_name(stage_name)[1]) for stage_name in self.architecture_config)
        nencoders = self.nstages - ndecoders
        assert nencoders == (ndecoders + 1), f"nencoders must be ndecoders + 1, but got nencoders={nencoders} and ndecoders={ndecoders}"
        
        # {height: size} for Upsample
        # example
        # 1: (128, 128)
        # 2: (64, 64)
        # 3: (32, 32)
        # 4: (16, 16)
        self.stages_size_map: Dict[int, Tuple[int, int]] = _stage_sizes_map(in_shape, self.height)  
        
        # ============================== loss stage weights ==============================
        
        if isinstance(loss_stage_weights, str):
            assert loss_stage_weights == HYPERSHPERE_U2NET_LOSS_WEIGHTS_UNIFORM, f"{loss_stage_weights} not implemented"
            self.hparams["loss_stage_weights_values"] = (1.0,) * (self.height + 1) 
        
        else:
            assert isinstance(loss_stage_weights, tuple), f"loss_stage_weights must be a tuple, but got {type(loss_stage_weights)}"
            assert all(isinstance(x, float) for x in loss_stage_weights), f"loss_stage_weights must be a tuple of floats, but got {loss_stage_weights}"
            assert len(loss_stage_weights) == (self.height + 1), f"loss_stage_weights must be a tuple of length {self.height + 1}, but got {loss_stage_weights} (len={len(loss_stage_weights)})"
            self.hparams["loss_stage_weights_values"] = loss_stage_weights
            
        # ============================== make layers ==============================
        
        for stage_name, (rsu_config, side_nchannels) in self.architecture_config.items():
            
            # parse everything            
            height_idx, isdecoder = parse_stage_name(stage_name)
            rsu_height, rsu_in_ch, rsu_mid_ch, rsu_out_ch, rsu_dilated, rsu_basic_block = rsu_config
            
            if isdecoder or height_idx == self.height:
                assert side_nchannels is not None, f"side channels must be specified for decoder RSU {rsu_name}"
            
            else: 
                assert side_nchannels is None, f"side channels must not be specified for encoder RSU {rsu_name}"
            
            # build rsu block
            self.add_module(stage_name, RSU(rsu_name(height_idx, isdecoder), *rsu_config))
            
            is_encoder_not_lowest_stage = not isdecoder and height_idx < self.height
            has_upsample = (isdecoder and height_idx > 1) or (not isdecoder and height_idx == self.height)
            
            if is_encoder_not_lowest_stage:
                self.add_module(
                    downsample_name(height_idx), 
                    # nn.MaxPool2d(2, stride=2, ceil_mode=True),
                    nn.Conv2d(
                        in_channels=rsu_out_ch, out_channels=rsu_out_ch, 
                        kernel_size=2, stride=2, 
                    )
                )
                
            elif has_upsample:
                side_upsample_factor = in_shape[0] // self.stages_size_map[height_idx][0]
                
                # about ConvTranspose2d upsample
                # say n is the factor of upsampling
                # then using kernel size = stride = n, padding = 0, and dilate = 1 
                # will upsample the image (w, h) to (w * n, h * n)
                
                self.add_module(
                    upsample_name(height_idx),
                    # nn.Upsample(size=upper_stage_size, mode='bilinear', align_corners=False,)
                    nn.ConvTranspose2d(
                        in_channels=rsu_out_ch, out_channels=rsu_out_ch,
                        kernel_size=2, stride=2, padding=0, output_padding=0, dilation=1,
                    )
                )
            
                # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> OPTION 1: BILINIEAR INTERPOLATION
                # this requires
                # torch.use_deterministic_algorithms(False)
                self.add_module(
                    score_upsample_name(height_idx),
                    nn.Upsample(size=in_shape, mode='bilinear', align_corners=False,)
                )
                # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> OPTION 2: TRANSPOSED CONVOLUTION
                # # attention: the **2 for the scores must come after the upsample when using this option
                # self.add_module(
                #     score_upsample_name(height_idx),
                #     nn.ConvTranspose2d(
                #         in_channels=1, out_channels=1,
                #         kernel_size=side_upsample_factor, stride=side_upsample_factor, padding=0, output_padding=0, dilation=1,
                #     )
                # )
                # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
            
            # build side layer
            if side_nchannels is not None:
                
                self.add_module(
                    side_name(height_idx), 
                    nn.Conv2d(rsu_out_ch, side_nchannels, 1, padding='same')
                )
                
                self.add_module(
                    side_finalconv_name(height_idx), 
                    nn.Conv2d(side_nchannels, 1, 1, padding='same', bias=True)
                )
                
        # it combines the scores from all the sides
        self.fuseconv = nn.Conv2d(self.height, 1, 1, bias=True)
            
    def forward(self, inputs: Tensor, return_all_stage_scores=False) -> Tensor:
        
        assert len(inputs.shape) == 4, f"inputs must be a 4D tensor, but got {inputs.shape}"
        assert inputs.shape[1] == 3, f"inputs must be a 3 channel tensor, but got {inputs.shape}"
        in_shape = self.hparams["in_shape"]
        assert inputs.shape[2:] == in_shape, f"inputs must be of shape {in_shape}, but got {inputs.shape}"
        
        # {height: upsampled score map}
        # height=0 is the fused score map
        score_maps = []
        
        # build the unet iteratively
        def recursive_unet(x, height_idx=1):
            
            encoder = getattr(self, stage_name(height_idx, isdecoder=False))
            
            if height_idx < self.height:
                
                x1 = encoder(x)
                
                # recursive call
                downsample = getattr(self, downsample_name(height_idx))
                x2 = recursive_unet(downsample(x1), height_idx=height_idx + 1)
                
                # merge the branches (from left (x1), and from below (x2))
                concatenated_branches = torch.cat((x2, x1), 1)
                
                decoder = getattr(self, stage_name(height_idx, isdecoder=True))
                x = decoder(concatenated_branches)
                
            else:
                x = encoder(x)
            
            sidelayer = getattr(self, side_name(height_idx))
            sidefinalconv = getattr(self, side_finalconv_name(height_idx))
            sidefeatures = sidelayer(x)
            sidescore = sidefinalconv(sidefeatures)
            
            # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> OPTION 1: BILINIEAR INTERPOLATION
            # this requires
            # torch.use_deterministic_algorithms(False)
            sidescore = sidescore ** 2 
            
            if height_idx > 1:
                upsample = getattr(self, upsample_name(height_idx))
                x = upsample(x)
                
                score_upsample = getattr(self, score_upsample_name(height_idx))
                sidescore = score_upsample(sidescore) 
            # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> OPTION 2: TRANSPOSED CONVOLUTION
            # # i moved the square to after the upsample because using the 
            # # transposed convolution makes negative values possible
            # if height_idx > 1:
            #     upsample = getattr(self, upsample_name(height_idx))
            #     x = upsample(x)
                
            #     score_upsample = getattr(self, score_upsample_name(height_idx))
            #     sidescore = score_upsample(sidescore) ** 2
            # # except for the score at the level 1 because it has no upsample
            # else:
            #     sidescore = sidescore ** 2
            # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
                
            score_maps.append(sidescore)
            
            return x
        
        recursive_unet(inputs)
        
        # the first to be appended in the recursive call is the deepest (lowest resolution) score map 
        score_maps.reverse()
        fused_score = self.fuseconv(torch.cat(score_maps, dim=1)) ** 2  # dim 1 = channel
        
        if return_all_stage_scores:
            score_maps.insert(0, fused_score)
            return torch.cat(score_maps, dim=1)  # dim 1 = channel
        
        else:
            return fused_score
        
    def configure_optimizers(self):
        
        # =================================== optimizer =================================== 
        optimizer_name = self.hparams['optimizer_name']
        assert optimizer_name in OPTIMIZER_CHOICES, f"optimizer '{optimizer_name}' unknown"
        
        lr = self.hparams['lr']
        weight_decay = self.hparams['weight_decay']
        
        if optimizer_name == OPTIMIZER_SGD:
            optimizer = optim.SGD(
                self.parameters(), 
                lr=lr, 
                weight_decay=weight_decay, 
                momentum=0.9, 
                nesterov=True,
            )
        
        elif optimizer_name == OPTIMIZER_ADAM:
            optimizer = optim.Adam(
                self.parameters(), 
                lr=lr, 
                weight_decay=weight_decay,
                # defaults in the paper and torch
                betas=(0.9, 0.999),
                eps=1e-8,
            )  
        
        else:
            raise NotImplementedError(f'Optimizer type {optimizer_name} not known.')
        
        # ================================ scheduler ================================
        
        scheduler_name = self.hparams['scheduler_name']
        assert scheduler_name in SCHEDULER_CHOICES, f"scheduler '{scheduler_name}' unknown"
        
        if scheduler_name is None:
            return [optimizer]
        
        scheduler_parameters = self.hparams['scheduler_parameters']
        
        if scheduler_name == SCHEDULER_LAMBDA:
            assert len(scheduler_parameters) == 1, 'lambda scheduler needs 1 parameter' 
            scheduler = self.configure_scheduler_lambda(optimizer, lr_exp_decrease_rate=scheduler_parameters[0])
            
        else:
            raise NotImplementedError(f'LR scheduler type {scheduler_name} not known.')
        
        return [optimizer], [scheduler] 
    
    def _common_step(self, batch, batch_idx, stage):
        
        loss_name = self.hparams["loss_name"]
        loss_scoremap_strategy = self.hparams["loss_scoremap_strategy"]
        
        # inputs \in [0, 1]^{N x C=3 x H x W}
        # labels \in {0, 1}^{N}
        # gtmaps \in {0, 1}^{N x 1 x H x W}
        inputs, labels, gtmaps = batch
        batchsize = inputs.shape[0]
        
        # score_maps \in {0, 1}^{N x !!!self.height + 1!!! x H x W}
        # the "+1" corresponds to the fused score map
        score_maps = self(inputs, return_all_stage_scores=True)
        
        if loss_scoremap_strategy == LOSS_U2NET_SCOREMAP_STRATEGY_ALLSIDES:
            # i sum the stage maps directly to avoid using too much memory
            w, h = self.hparams["in_shape"]
            # total_loss_map \in R+^{N x 1 x H x W}
            loss_maps = torch.zeros((batchsize, 1, h, w), dtype=inputs.dtype, device=inputs.device)
            loss_stage_weights_values = torch.tensor(self.hparams['loss_stage_weights_values'], device=inputs.device)
            
            for map_idx in range(self.height + 1):
                # map_idx = 0 is the fused score map
                # loss_map \in R+^{N x 1 x H x W}
                stage_loss_maps, _ = self.hsc_loss(
                    score_map=score_maps[:, map_idx, :, :].unsqueeze(1), 
                    masks=gtmaps, 
                    labels=labels, 
                    loss_version=loss_name,
                )
                weight = loss_stage_weights_values[map_idx]
                loss_maps += weight * stage_loss_maps
                
            # loss \in R+
            # mean: per image and channel / per image channel mean / same on the whole batch 
            # loss = total_loss_map.mean(dim=(-2, -1)).mean(dim=-1).mean(dim=-1)
            # equivalently
            loss = loss_maps.mean()
            
        else:
            raise NotImplementedError(f'Loss scoremap strategy {loss_scoremap_strategy} not implemented.')
                
        # separate normal/anomaly
        with torch.no_grad():
            batch_pixel_count_normal = (gtmaps == 0).sum()
            batch_pixel_count_anomalous = (gtmaps == 1).sum().nan_to_num(0)
            
            fusescore_maps = score_maps[:, 0, :, :].unsqueeze(1)
            fusescore_mean = fusescore_maps.mean()
            fusescore_normals_mean = fusescore_maps[gtmaps == 0].mean()
            # the .nan_to_num(0.) is needed to avoid nans in the log because the 
            # [gtmaps == 1] may cause empty tensors
            # this will be 0 if there are no anomalies, which is the desired behavior 
            # although it will cause some false reduced value at the end of the epoch
            fusescore_anomalous_mean = fusescore_maps[gtmaps == 1].mean().nan_to_num(0.)
            
            loss_normals_mean = loss_maps[gtmaps == 0].mean()
            # the .nan_to_num(0.) is needed to avoid nans in the log because the 
            # [gtmaps == 1] may cause empty tensors
            # this will be 0 if there are no anomalies, which is the desired behavior 
            # although it will cause some false reduced value at the end of the epoch
            loss_anomalous_mean = loss_maps[gtmaps == 1].mean().nan_to_num(0.)

        self.log(f"{stage}/pixel-count-normal", batch_pixel_count_normal, on_step=False, on_epoch=True, reduce_fx="sum")
        self.log(f"{stage}/pixel-count-anomalous", batch_pixel_count_anomalous, on_step=False, on_epoch=True, reduce_fx="sum")
        
        self.log(f"{stage}/score-mean", fusescore_mean, on_step=False, on_epoch=True)
        self.log(f"{stage}/score-normals-mean", fusescore_normals_mean, on_step=False, on_epoch=True)
        self.log(f"{stage}/score-anomalous-mean", fusescore_anomalous_mean, on_step=False, on_epoch=True)
        
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
            score_maps=fusescore_maps,
            # outputs
            loss=loss,
        )
    
    def _common_epoch_end(self, outputs, stage):
        self.last_epoch_outputs = merge_steps_outputs(outputs)
            
    def training_step(self, batch, batch_idx):    
        return self._common_step(batch, batch_idx, stage=RunningStage.TRAINING)
    
    def training_epoch_end(self, training_steps_outputs) -> None:
        return self._common_epoch_end(training_steps_outputs, stage=RunningStage.TRAINING)
    
    def validation_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, stage=RunningStage.VALIDATING)
    
    def validation_epoch_end(self, validation_steps_outputs):
        return self._common_epoch_end(validation_steps_outputs, stage=RunningStage.VALIDATING)
        
    def test_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, stage=RunningStage.TESTING)
    
    def test_epoch_end(self, test_step_outputs):
        return self._common_epoch_end(test_step_outputs, stage=RunningStage.TESTING)
    
    def teardown(self, stage=None):
        self.last_epoch_outputs = None

