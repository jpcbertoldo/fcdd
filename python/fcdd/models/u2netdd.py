from typing import Dict, Tuple
from fcdd.models.bases import FCDDNet
import torch
import torch.nn as nn

import math

__all__ = ['U2NET_full', 'U2NET_lite']


def _upsample_like(x, size):
    return nn.Upsample(size=size, mode='bilinear', align_corners=False)(x)


def _size_map(x, height):
    """
    {height: size} for Upsample
    """
    
    # x.shape [batch_size, channels, height, width]
    img_shape = x.shape[-2:]  # [height, width]
    
    sizes = {
        1: tuple(img_shape),
    }
    
    for h in range(2, height):
        previous_size = sizes[h - 1]
        sizes[h] = tuple(math.ceil(w / 2) for w in previous_size)
        
    return sizes


class REBNCONV(nn.Module):
    
    def __init__(self, in_ch=3, out_ch=3, dilate=1):
        super(REBNCONV, self).__init__()

        self.conv_s1 = nn.Conv2d(in_ch, out_ch, 3, padding=1 * dilate, dilation=1 * dilate)
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
        
        if in_ch == out_ch:
            self.bn_id = nn.BatchNorm2d(out_ch)
        else:
            self.bn_id = None
        
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        x_33 = self.bn_33(self.conv_33(x))
        x_11 = self.bn_11(self.conv_11(x))
        x_id = self.bn_id(x) if self.bn_id is not None else None
        summed = (
            x_33 + x_11 + x_id
            if x_id is not None else
            x_33 + x_11
        )
        return self.relu(summed)


BASIC_BLOCK_REBNCONV_NAME = "rebnconv"
BASIC_BLOCK_REPVGG_NAME = "repvgg"
BASIC_BLOCK_NAMES = (BASIC_BLOCK_REBNCONV_NAME, BASIC_BLOCK_REPVGG_NAME,)


class RSU(nn.Module):
    
    def __init__(self, name, height, in_ch, mid_ch, out_ch, dilated=False, basic_block=BASIC_BLOCK_REBNCONV_NAME):
        super(RSU, self).__init__()
        self.name = name
        self.height = height
        self.dilated = dilated
        assert basic_block in BASIC_BLOCK_NAMES
        self.basic_block = basic_block
        if basic_block == BASIC_BLOCK_REBNCONV_NAME:
            self.basic_block_class = REBNCONV
        elif basic_block == BASIC_BLOCK_REPVGG_NAME:
            self.basic_block_class = RepVGGConv
        else:
            raise ValueError(f"basic_block must be one of {BASIC_BLOCK_NAMES}")
        self._make_layers(height, in_ch, mid_ch, out_ch, dilated)

    def forward(self, x):
        sizes = _size_map(x, self.height)
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


CONFIG_HEIGHT6_FULL = {
    # cfgs for building RSUs and sides
    # {stage : [name, (height(L), in_ch, mid_ch, out_ch, dilated), side]}
    'stage1': ('En_1', (7, 3, 32, 64), None),
    'stage2': ('En_2', (6, 64, 32, 128), None),
    'stage3': ('En_3', (5, 128, 64, 256), None),
    'stage4': ('En_4', (4, 256, 128, 512), None),
    'stage5': ('En_5', (4, 512, 256, 512, True), None),
    'stage6': ('En_6', (4, 512, 256, 512, True), 512),
    'stage5d': ('De_5', (4, 1024, 256, 512, True), 512),
    'stage4d': ('De_4', (4, 1024, 128, 256), 256),
    'stage3d': ('De_3', (5, 512, 64, 128), 128),
    'stage2d': ('De_2', (6, 256, 32, 64), 64),
    'stage1d': ('De_1', (7, 128, 16, 64), 64),
}

CONFIG_HEIGHT6_LITE = {
    # cfgs for building RSUs and sides
    # {stage : [name, (height(L), in_ch, mid_ch, out_ch, dilated), side]}
    'stage1': ('En_1', (7, 3, 16, 64), None),
    'stage2': ('En_2', (6, 64, 16, 64), None),
    'stage3': ('En_3', (5, 64, 16, 64), None),
    'stage4': ('En_4', (4, 64, 16, 64), None),
    'stage5': ('En_5', (4, 64, 16, 64, True), None),
    'stage6': ('En_6', (4, 64, 16, 64, True), 64),
    'stage5d': ('De_5', (4, 128, 16, 64, True), 64),
    'stage4d': ('De_4', (4, 128, 16, 64), 64),
    'stage3d': ('De_3', (5, 128, 16, 64), 64),
    'stage2d': ('De_2', (6, 128, 16, 64), 64),
    'stage1d': ('De_1', (7, 128, 16, 64), 64),
}

CONFIG_HEIGHT4_LITE = {
    # cfgs for building RSUs and sides
    # {stage : [name, (height(L), in_ch, mid_ch, out_ch, dilated), side]}
    'stage1': ('En_1', (7, 3, 16, 64, False,), None),
    'stage2': ('En_2', (6, 64, 16, 64, False,), None),
    'stage3': ('En_3', (5, 64, 16, 64, False,), None),
    'stage4': ('En_4', (4, 64, 16, 64, False,), 64),
    'stage3d': ('De_3', (5, 128, 16, 64, False,), 64),
    'stage2d': ('De_2', (6, 128, 16, 64, False,), 64),
    'stage1d': ('De_1', (7, 128, 16, 64, False,), 64),
}

CONFIG_HEIGHT4_LITE_REPVGG = {
    # cfgs for building RSUs and sides
    # {stage : [name, (height(L), in_ch, mid_ch, out_ch, dilated), side]}
    'stage1': ('En_1', (7, 3, 16, 64, False, BASIC_BLOCK_REPVGG_NAME), None),
    'stage2': ('En_2', (6, 64, 16, 64, False, BASIC_BLOCK_REPVGG_NAME), None),
    'stage3': ('En_3', (5, 64, 16, 64, False, BASIC_BLOCK_REPVGG_NAME), None),
    'stage4': ('En_4', (4, 64, 16, 64, False, BASIC_BLOCK_REPVGG_NAME), 64),
    'stage3d': ('De_3', (5, 128, 16, 64, False, BASIC_BLOCK_REPVGG_NAME), 64),
    'stage2d': ('De_2', (6, 128, 16, 64, False, BASIC_BLOCK_REPVGG_NAME), 64),
    'stage1d': ('De_1', (7, 128, 16, 64, False, BASIC_BLOCK_REPVGG_NAME), 64),
}


# LAYER NAMING CONVENTIONS

def rsu_name2height(rsu_name: str) -> int:
    h = int(rsu_name.split("_")[-1])
    assert 0 < h < 10 
    return h


def rsu_side_name(height_: int) -> str:
    return f'side{height_}'


def stage_name(height_: int, decoder_: bool) -> str:
    return f'stage{height_}{"d" if decoder_ else ""}'


OUTCONV_NAME = 'outconv'
DOWNSAMPLE_NAME = 'downsample'


class _U2NETDD(FCDDNet):
    
    def __init__(self, in_shape: Tuple[int, int, int], config, bias=False):
        
        # these two are properties in BaseNet, they should be uncommented later
        # self.in_shape = in_shape
        # self.bias = bias
        
        super(_U2NETDD, self).__init__(in_shape=in_shape, bias=bias)
        
        self.config = config
        
        # ============================== make layers ==============================
        self.nstages = len(self.config)
        assert self.nstages % 2 == 1, f"nstages in the config must be odd, but got {self.nstages}"
        
        self.height = int((self.nstages + 1) / 2)
        assert self.height > 1, f"height must be greater than 1, but got {self.height}"
        
        self.add_module(DOWNSAMPLE_NAME, nn.MaxPool2d(2, stride=2, ceil_mode=True))
        
        for stage_name, (rsu_name, rsu_config, rsu_side_nchannels) in self.config.items():
            
            # build rsu block
            self.add_module(stage_name, RSU(rsu_name, *rsu_config))
            
            # build side layer
            if rsu_side_nchannels is not None:
                self.add_module(
                    rsu_side_name(rsu_name2height(rsu_name)), 
                    nn.Conv2d(rsu_side_nchannels, 1, 1, padding=1, bias=self.bias)
                )
                
        # build fuse layer
        self.add_module(OUTCONV_NAME, nn.Conv2d(int(self.height), 1, 1, bias=self.bias))
        
    def forward(self, inputs):
        
        # {height: size} for Upsample
        sizes: Dict[int, Tuple[int, int]] = _size_map(inputs, self.height)  
        
        maps = []  # storage for maps

        # build the unet iteratively
        def unet(x, height_=1):
            
            encode = getattr(self, stage_name(height_, decoder_=False))
            
            downsample = getattr(self, DOWNSAMPLE_NAME)
            
            if height_ < self.height:
                
                x1 = encode(x)
                
                # recursive call
                x2 = unet(downsample(x1), height_=height_ + 1)
                
                # merge the branches (from left (x1), and from below (x2))
                concatenated_branches = torch.cat((x2, x1), 1)
                
                decode = getattr(self, stage_name(height_, decoder_=True))
                x = decode(concatenated_branches)
                
            else:
                x = encode(x)
                
            # side output saliency map (before sigmoid)
            xside = getattr(self, rsu_side_name(height_))(x)
            xside = nn.Upsample(
                size=sizes[1], 
                mode='bilinear', 
                align_corners=False,
            )(xside)
            
            maps.append(xside)

            x = (
                nn.Upsample(
                    size=sizes[height_ - 1],  # size from the height above (higher res)
                    mode='bilinear', 
                    align_corners=False,
                )(x)
                if height_ > 1 else 
                x
            )
            
            return x

        unet(inputs)
        
        # fuse saliency probability maps
        maps.reverse()
        fuseconv = getattr(self, OUTCONV_NAME)
        fused = fuseconv(torch.cat(maps, 1))
        maps.insert(0, fused)
        
        return maps

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
                    

class U2NETDD_HEIGHT4_LITE(_U2NETDD):
    
    def __init__(self, in_shape: Tuple[int, int, int], bias=False, ):
        super().__init__(in_shape=in_shape, bias=bias, config=CONFIG_HEIGHT4_LITE)


if __name__ == '__main__':
    model = U2NETDD_HEIGHT4_LITE(
        in_shape=(3, 64, 64),
        bias=True,
    )
    print(model)
    inputs = torch.zeros(16, 3, 64, 64)
    print(f"inputs shape: {inputs.shape}")
    outputs = model(inputs)
    print(f"outputs shape: type={type(outputs)} len={len(outputs)}")
    for out in outputs:
        print(f"out shape: {out.shape}")
