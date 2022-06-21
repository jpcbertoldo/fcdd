from typing import Dict
from fcdd.models.bases import FCDDNet
import torch
import torch.nn as nn
import os.path as pt
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from fcdd.models.bases import FCDDNet
from torch.hub import load_state_dict_from_url
import torch.nn as nn


class UNETDD:
    pass

class UNETDD_VGG_11BN(FCDDNet, UNETDD):
    
    # VGG_11BN based net with most of the VGG layers having weights 
    # pretrained on the ImageNet classification task.
    
    def __init__(self, in_shape, **kwargs):
        
        super().__init__(in_shape, **kwargs)
        assert self.bias, 'VGG net is only supported with bias atm!'
        
        state_dict = load_state_dict_from_url(
            torchvision.models.vgg.model_urls['vgg11_bn'],
            model_dir=pt.join(pt.dirname(__file__), '..', '..', '..', 'data', 'models')
        )
        features_state_dict = {k[9:]: v for k, v in state_dict.items() if k.startswith('features')}

        # all frozen!
        self.features = nn.Sequential(
            # conv1, res 1
            nn.Conv2d(3, 64, 3, 1, 1),  # 0
            nn.BatchNorm2d(64),  # 1
            nn.LeakyReLU(True),  # 2
            nn.MaxPool2d(2, 2),  # 3
            # conv2, res 2
            nn.Conv2d(64, 128, 3, 1, 1),  # 4
            nn.BatchNorm2d(128),  # 5
            nn.LeakyReLU(True),  # 6
            nn.MaxPool2d(2, 2),  # 7
            # conv3, res 3
            nn.Conv2d(128, 256, 3, 1, 1),  # 8
            nn.BatchNorm2d(256),  # 9
            nn.LeakyReLU(True),  # 10
            nn.Conv2d(256, 256, 3, 1, 1),  # 11
            nn.BatchNorm2d(256),  # 12
            nn.LeakyReLU(True),  # 13
            nn.MaxPool2d(2, 2),  # 14
            # conv5, res 4
            nn.Conv2d(256, 512, 3, 1, 1),  # 15
            nn.BatchNorm2d(512),  # 16
            nn.LeakyReLU(True),  # 17
            nn.Conv2d(512, 512, 3, 1, 1),  # 18
            nn.BatchNorm2d(512),  # 19
            nn.LeakyReLU(True),  # 20
            # CUT 
            nn.MaxPool2d(2, 2),  # 21
            # res 5
            nn.Conv2d(512, 512, 3, 1, 1),  # 22
            nn.BatchNorm2d(512),  # 23
            nn.LeakyReLU(True),  # 24
            nn.Conv2d(512, 512, 3, 1, 1),  # 25
            nn.BatchNorm2d(512),  # 26
            nn.LeakyReLU(True),  # 27
            nn.MaxPool2d(2, 2)  # 28
        )
        
        for m in self.features:
            for p in m.parameters():
                p.requires_grad = False
        self.features.load_state_dict(features_state_dict)
        self.features = self.features[:-8]
        
        self.internal_features = {}
        def get_activation(name):
            def hook(model, input, output):
                self.internal_features[name] = output
            return hook

        self.features[2].register_forward_hook(get_activation('res01.feat01'))
        self.features[6].register_forward_hook(get_activation('res02.feat01'))
        self.features[10].register_forward_hook(get_activation('res03.feat01'))
        self.features[13].register_forward_hook(get_activation('res03.feat02'))
        self.features[17].register_forward_hook(get_activation('res04.feat01'))
        self.features[20].register_forward_hook(get_activation('res04.feat02'))
        
        self.conv0101 = nn.Conv2d(64, 16, 1, 1, 0)
        self.conv0201 = nn.Conv2d(128, 16, 1, 1, 1)
        self.conv0301 = nn.Conv2d(256, 16, 1, 1, 1)
        self.conv0302 = nn.Conv2d(256, 16, 1, 1, 1)
        self.conv0401 = nn.Conv2d(512, 16, 1, 1, 1)
        self.conv0402 = nn.Conv2d(512, 16, 1, 1, 1)

        self.upsample = nn.Upsample(
            size=self.in_shape[-1], mode='bilinear', align_corners=False,
        )
        
        self.fuse = nn.Sequential(
            nn.Conv2d(96, 1, 1, 1, 0, bias=True),
            # nn.BatchNorm2d(64), 
            # nn.LeakyReLU(True), 
            # nn.Conv2d(64, 1, 1, 1, 0, bias=True),  
        )

    def forward(self, x, return_all_maps=True):
        self.internal_features = {}
        x = self.features(x)
        
        score0101 = self.conv0101(self.internal_features["res01.feat01"])
        score0201 = self.upsample(self.conv0201(self.internal_features["res02.feat01"]))
        score0301 = self.upsample(self.conv0301(self.internal_features["res03.feat01"]))
        score0302 = self.upsample(self.conv0302(self.internal_features["res03.feat02"]))
        score0401 = self.upsample(self.conv0401(self.internal_features["res04.feat01"]))
        score0402 = self.upsample(self.conv0402(self.internal_features["res04.feat02"]))
        
        scores = [score0101, score0201, score0301, score0302, score0401, score0402]
        score_fused = self.fuse(torch.cat(scores, dim=1))
        return score_fused
        
        scores.insert(0, score_fused)
        scores = torch.cat(scores, dim=1)
        
        if return_all_maps:
            return scores

        else:
            return score_fused

    @property
    def reception(self):
        """
        in ReceptiveNet
        delegated to ReceptiveModule:
        return {'n': self._n, 'j': self._j, 'r': self._r, 's': self._s, 'img_shape': self._in_shape}
        mocked by
        """
        return {'n': 0, 'j': 0, 'r': 0, 's': 0, 'img_shape': self.in_shape}
    
    @property
    def initial_reception(self):
        """
        in ReceptiveNet
        defined at ReceptiveNet.__init__
        using ReceptiveModule.set_reception
        mocked by
        """
        return {'n': 0, 'j': 0, 'r': 0, 's': 0, 'img_shape': self.in_shape}
    
    def reset_parameters(self):
        """
        uses Module.apply (apply a func to all submodules)
        with ReceiverModule.__weight_reset
        mocked by
        """
        pass
    
    def _create_conv2d(self, *args, **kwargs):
        """called by children of ReceptiveNet, not used here, should not be used"""
        raise Exception("_create_conv2d should not be used")
    
    def _create_maxpool2d(self, *args, **kwargs):
        """called by children of ReceptiveNet, not used here, should not be used"""
        raise Exception("_create_maxpool2d should not be used")
    
    def set_reception(self, *args, **kwargs):
        """called by ReceptiveModule, mocked by"""
        pass
    
    def receptive_upsample(self, pixels, *args, **kwargs):
        """
        u2net doesnt need upsampling
        just return the input
        """
        return pixels
    
    @property
    def device(self):
        """this shoulnd be necessary"""
        raise Exception("device should not be used")
    
    def __upsample_nn(self, pixels):
        """this shoulnd be necessary"""
        raise Exception("__upsample_nn should not be used")
    
    def get_grad_heatmap(self, *args, **kwargs):
        """this shoulnd be necessary"""
        raise Exception("get_grad_heatmap should not be used")
                    


if __name__ == '__main__':
    model = UNETDD_VGG_11BN(in_shape=(3, 64, 64), bias=True)
    print(model)
    inputs = torch.zeros(16, 3, 64, 64)
    print(f"inputs shape: {inputs.shape}")
    outputs = model(inputs)
    print(f"outputs shape: type={type(outputs)} len={len(outputs)}")
    for out in outputs:
        print(f"out shape: {out.shape}")
        
        
        

# self.features = nn.Sequential(OrderedDict([
#     # block 01, res 1
#     ("block01-conv01", nn.Conv2d(3, 64, 3, 1, 1),),
#     ("block01-normalization01", nn.BatchNorm2d(64),),
#     ("block01-activation01", nn.LeakyReLU(True),),
#     ("block01-downsample01", nn.MaxPool2d(2, 2),),
#     # block 02, res 1/2
#     ("block02-conv01", nn.Conv2d(64, 128, 3, 1, 1),),
#     ("block02-normalization01", nn.BatchNorm2d(128),),
#     ("block02-activation01", nn.LeakyReLU(True),),
#     ("block02-downsample01", nn.MaxPool2d(2, 2),),
#     # block 03, res 1/4
#     ("block03-conv01", nn.Conv2d(128, 256, 3, 1, 1),),
#     ("block03-normalization01", nn.BatchNorm2d(256),),
#     ("block03-activation01", nn.LeakyReLU(True),),
#     ("block03-conv02", nn.Conv2d(256, 256, 3, 1, 1),),
#     ("block03-normalization02", nn.BatchNorm2d(256),),
#     ("block03-activation02", nn.LeakyReLU(True),),
#     ("block03-downsample01", nn.MaxPool2d(2, 2),),
#     # block 04, res 1/8
#     ("block04-conv01", nn.Conv2d(256, 512, 3, 1, 1),),
#     ("block04-normalization01", nn.BatchNorm2d(512),),
#     ("block04-activation01", nn.LeakyReLU(True),),
#     ("block04-conv02", nn.Conv2d(512, 512, 3, 1, 1),),
#     ("block04-normalization02", nn.BatchNorm2d(512),),
#     ("block04-activation02", nn.LeakyReLU(True),),
#     # CUT 
#     ("block04-downsample01", nn.MaxPool2d(2, 2),),
#     # block 05, res 1/16
#     ("block05-conv01", nn.Conv2d(512, 512, 3, 1, 1),),
#     ("block05-normalization01", nn.BatchNorm2d(512),),
#     ("block05-activation01", nn.LeakyReLU(True),),
#     ("block05-conv02", nn.Conv2d(512, 512, 3, 1, 1),),
#     ("block05-normalization02", nn.BatchNorm2d(512),),
#     ("block05-activation02", nn.LeakyReLU(True),),
#     ("block05-downsample01", nn.MaxPool2d(2, 2),),
# ]))