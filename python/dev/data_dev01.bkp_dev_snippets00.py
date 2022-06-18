# In[]        

import copy
import functools
from collections import Counter
from collections.abc import Sequence
from time import time
from turtle import forward
from typing import Callable, List, Tuple

import matplotlib
import numpy as np
import torch
import torchvision.transforms.transforms as T
import torchvision.transforms.functional as TF
import torchvision.transforms.functional_tensor as TFT
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from torch.utils.data import DataLoader
from torch import Tensor

# In[]

# snippet for testing numpy random generators

import numpy, json, copy

seed1, seed2, seed3 = 0, 0, 1
g1 = numpy.random.Generator(numpy.random.PCG64(numpy.random.SeedSequence(seed1)))
g2 = numpy.random.Generator(numpy.random.PCG64(numpy.random.SeedSequence(seed2)))
g3 = numpy.random.Generator(numpy.random.PCG64(numpy.random.SeedSequence(seed3)))

s11 = json.dumps(g1.bit_generator.state)
s21 = json.dumps(g2.bit_generator.state)
s31 = json.dumps(g3.bit_generator.state)

assert s11 == s21
assert s11 != s31

assert g1.random() == g2.random()

s12 = json.dumps(g1.bit_generator.state)
s22 = json.dumps(g2.bit_generator.state)

assert s12 == s22

g3.bit_generator.state = copy.deepcopy(g1.bit_generator.state)
assert g1.random() == g3.random()


# In[]


class C:
    n = None
    def fun(self, x):
        self.n = x

c1 = C()
c2 = C()

assert c1.n is None
assert c2.n is None

c1.fun(5)

assert c1.n == 5
assert c2.n is None


# In[]

"""
this snippet will cut crops of the same size in a batch
where each instance can have different positions

the global idea is that one must first cut on the height direction
then on the width direction (transpose(-2, -1) at the end), and to 
deal with the channels we transpose(0, 1) 

good luck to decrypt this function :)
but this snippet should convince you that it works
"""

# args
batch = torch.arange(3*4*5).view(3, 1, 4, 5).repeat(1, 2, 1, 1)
batch[:, 0, :, :] *= 1
batch[:, 1, :, :] *= 10
batch
top = torch.tensor([0, 0, 2])
left = torch.tensor([0, 3, 3])
width = 2
height = 2

(batchsize, nchannels, im_height, im_width) = batchshape = batch.shape
croped_batchshape = batchshape[:-2] + (width, height)

right = left + width
bottom = top + height

ground_truth_select = torch.zeros_like(batch, dtype=bool)
ground_truth_select[0, ..., :2, :2] = True
ground_truth_select[1, ..., :2, -2:] = True
ground_truth_select[2, ..., -2:, -2:] = True
ground_truth_croped_batch = batch[ground_truth_select].reshape(croped_batchshape)

height_crop = torch.arange(im_height).expand(batchsize, im_height)
height_crop = torch.logical_and(
    top.view(-1, 1) <= height_crop, 
    height_crop < bottom.view(-1, 1),
)
height_crop.shape

width_crop = torch.arange(im_width).expand(batchsize, im_width)
width_crop = torch.logical_and(
    left.view(-1, 1) <= width_crop, 
    width_crop < right.view(-1, 1),
)
width_crop.shape

crop_shape = batchshape[:-2] + (height, width)

batch.transpose(0, 1)\
    [:, height_crop, :]\
    .view((nchannels, batchsize) + (height, im_width))\
    .transpose(-2, -1)\
    [:, width_crop, :]\
    .view((nchannels, batchsize) + (width, height))\
    .transpose(-2, -1)\
    .transpose(0, 1)
