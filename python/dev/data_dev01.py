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


"""
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
"""


"""
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
"""


"""
this snippet will cut crops of the same size in a batch
where each instance can have different positions

the global idea is that one must first cut on the height direction
then on the width direction (transpose(-2, -1) at the end), and to 
deal with the channels we transpose(0, 1) 

good luck to decrypt this function :)
but this snippet should convince you that it works

# args
batch = torch.arange(3*4*5).view(3, 1, 4, 5).repeat(1, 2, 1, 1)
batch[:, 0, :, :] *= 1
batch[:, 1, :, :] *= 10
batch
top = torch.tensor([0, 0, 2])
left = torch.tensor([0, 3, 3])
width = 2
height = 2
# todo make width/height int args and validate right/bottom in generation

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
"""

ANOMALY_TARGET = 1
NOMINAL_TARGET = 0


# =============================================== TRANSFORM MIXINS ===============================================


class NumpyRandomTransformMixin:
    """
    Mixin that will get a numpy.random.Generator from the init kwargs and make sure it is valid.
    """
    
    def _grab_generator_from_kwargs(self, generator: np.random.Generator):
        assert generator is not None, "generator must be provided as a kwarg"
        assert isinstance(generator, np.random.Generator), f"generator must be a numpy.random.Generator, got {type(generator)}"
        self.generator = generator
        
    def _init_generator(__init__):
        """Make sure the generator is given as a kwarg at the init function."""
        
        @functools.wraps(__init__)
        def wrapper(self, *args, **kwargs):
            assert "generator" in kwargs, "generator must be provided"
            self._grab_generator_from_kwargs(kwargs.pop("generator"))
            __init__(self, *args, **kwargs)
        
        return wrapper


class BatchTransformMixin:
    """
    Make sure that the argument given to the transform is a batch by checking the number of dimensions.
    Use @_validate_before_and_after on top of __call__ to make sure that the result is a batch as well.
    """
    
    def _validate_batch(self, batch: Tensor):
        assert isinstance(batch, Tensor), f"batch must be a Tensor, got {type(batch)}"
        assert batch.ndimension() == 4, f"batch must be a batch of images (expected to have dimensions [B, C, H, W]), got {batch.ndimension()} dimensions"
    
    def _validate_before_and_after(__call__: Callable[[Tensor], Tensor]):
        
        @functools.wraps(__call__)
        def wrapper(self, batch: Tensor, **kwargs) -> Tensor:
            self._validate_batch(batch)
            batchsize_before = batch.size(0)
            batch = __call__(self, batch, **kwargs)
            self._validate_batch(batch)
            batchsize_after = batch.size(0)
            assert batchsize_before == batchsize_after, f"batchsize before and after transform must be the same, got {batchsize_before} and {batchsize_after}"
            return batch
        
        return wrapper
    
    
class MultiBatchTransformMixin:
    """Make sure to always have the same number of batches and that they all have the same batch size."""
    
    nbatches = None
    
    def _validate_batches(self, *batches: List[Tensor]):
        nbatches = len(batches)
        if self.nbatches is None:
            self.nbatches = nbatches
        assert self.nbatches == nbatches, f"number of batches must always be the same, got {self.nbatches} and {nbatches}"
        for batch in batches:
            assert isinstance(batch, Tensor), f"batch must be a Tensor, got {type(batch)}"
            assert batch.ndimension() == 4, f"batch must be a batch of images (expected to have dimensions [B, C, H, W]), got {batch.ndimension()} dimensions"
        batchsizes = np.array([batch.size(0) for batch in batches])
        assert np.all(batchsizes == batchsizes[0]), f"all batches must have the same batch size, got {batchsizes}"

    def _validate_before_and_after(__call__: Callable[[Tensor], Tensor]):
        
        @functools.wraps(__call__)
        def wrapper(self, *batches: List[Tensor], **kwargs) -> List[Tensor]:
            self._validate_batches(*batches)
            batch0 = batches[0]
            batchsize_before = batch0.size(0)
            batches = __call__(self, *batches, **kwargs)
            self._validate_batches(*batches)
            batchsize_after = batch0.size(0)
            assert batchsize_before == batchsize_after, f"batchsize before and after transform must be the same, got {batchsize_before} and {batchsize_after}"
            return batches
        
        return wrapper
    

class RandomMultiTransformMixin:
    """
    Make sure that all aruments are called with the same random state, so the same transform is applied to all images or batches.
    This mixin does NOT assume that the inputs are batches.
    """
    
    def _use_same_random_state_on_all_args(__call__: Callable[[Tensor], Tensor]):
        
        @functools.wraps(__call__)
        def wrapper(self, *args: List[Tensor], **kwargs) -> List[Tensor]:
            assert isinstance(self, NumpyRandomTransformMixin), f"only use this decorator if {self} is a NumpyRandomTransformMixin"
            initial_state = copy.deepcopy(self.generator.bit_generator.state)
            returns = []
            for arg in args:
                self.generator.bit_generator.state = copy.deepcopy(initial_state)
                # the [arg] is so that the interface of call (which should have *args) is preserved
                # same logic for the [0] afterwards
                ret = __call__(self, [arg], **kwargs)
                returns.append(ret[0])
                # by setting the random state at the beginning, the state will still have
                # advanced as this loop finishes the last iteration
            return returns
        
        return wrapper


class TransformsMixin:
    """The transform should get a list of transforms at the init."""
    
    def _grab_transforms_from_kwargs(self, transforms: Sequence[Callable]):
        assert transforms is not None, "transforms must be provided"
        assert isinstance(transforms, Sequence), f"transforms must be a sequence, got {type(transforms)}"
        assert len(transforms) > 0, "transforms must not be empty"
        for idx, transform in enumerate(transforms):
            assert callable(transform), f"transforms must be callable, got {transform} at index {idx}"
        self.transforms = transforms
        
    def _init_transforms(__init__):
        """Make sure the transforms is given as a kwarg at the init function."""
        
        @functools.wraps(__init__)
        def wrapper(self, *args, **kwargs):
            assert "transforms" in kwargs, "transforms must be provided as a kwwarg"
            self._grab_transforms_from_kwargs(kwargs.pop("transforms"))
            __init__(self, *args, **kwargs)
        
        return wrapper


# =============================================== RANDOM CHOICE ===============================================


def _apply_chosen_batch_transforms(
    batch: Tensor, 
    thresholds: np.ndarray, 
    probabilities: np.ndarray, 
    transforms: Sequence[Callable]
) -> Tensor:

    assert probabilities.ndim == 1, f"got {probabilities.ndim}"
    assert thresholds.ndim == 1, f"got {thresholds.ndim}"
    assert batch.shape[0] == probabilities.shape[0], f"got {batch.shape[0], probabilities.shape[0]}"
    ntransforms = len(transforms)
    assert ntransforms == thresholds.shape[0], f"got {ntransforms} transforms and {thresholds.shape[0]} thresholds"
    
    # probabilities are between 0 and 1
    assert (0 <= probabilities < 1).all(), f"got probabilities {probabilities}, they must be between 0 and 1"
    assert (0 <= thresholds < 1).all(), f"got thresholds {thresholds}, they must be between 0 and 1"
    
    # the thresholds are sorted and cannot be the same (< instead of <=)
    assert (thresholds[:-1] < thresholds[1:]).all(), f"got thresholds {thresholds}, they must be sorted"
    
    # the reshapes + ">" will make a 2d table with true/false 
    # where "true" means the probability at line i is higher than the threshold at column j
    # so the sum(axis=-1) is counting how many thresholds the proba is higher than
    # the -1 makes it a 0-starting index
    selected_ops_indices = (probabilities.reshape(-1, 1) > thresholds.reshape(1, -1)).sum(axis=-1) - 1
    
    # idx: ndarray
    # the idx is the index of the transform in self.transforms that will be applied to the
    # group of instances whose indices are in ndarray
    for idx, transf in enumerate(transforms):
        select = selected_ops_indices == idx
        batch[select] = transf(batch[select])
        
    return batch


class BatchRandomChoice(NumpyRandomTransformMixin, BatchTransformMixin, TransformsMixin):
    """
    Pick a transformation randomly picked from a list for each instance in the batch. 
    This transform does not support torchscript.
    Use probabilities intervals between 0 and 1 to decide which operation to use.
    Ex: if there are 4 operations, define 4 thresholds: 0, .25, .50, .75; 
        draw a sample from a uniform distribution before 0 and 1: .33 (e.g.)
        0. < .25 < .33* 
        .33 is bigger than 2 thresholds so the second operation (index=1)    
    """
    
    @NumpyRandomTransformMixin._init_generator
    @TransformsMixin._init_transforms
    def __init__(self, *args, **kwargs):
        self.ntransforms = len(self.transforms)
        self._thresholds = np.linspace(0, 1, self.ntransforms, endpoint=False)
        
        for idx, transf in enumerate(self.transforms):
            assert isinstance(transf, BatchTransformMixin), f"transforms must be BatchTransformMixin, got {transf} at index {idx}"
            
    @BatchTransformMixin._validate_before_and_after
    def __call__(self, batch: Tensor) -> Tensor:
        # keep all the randomness separate from the transform logic
        batchsize = batch.shape[0]
        probabilities = np.random.uniform(low=0, high=1, size=batchsize)
        return _apply_chosen_batch_transforms(batch, self._thresholds, probabilities, self.transforms)


def _apply_chosen_multibatch_transforms(
    *batches: List[Tensor], 
    # the nones are just to make them kwargs
    thresholds: np.ndarray = None, 
    probabilities: np.ndarray = None, 
    transforms: Sequence[Callable] = None
) -> List[Tensor]:

    assert thresholds is not None, "thresholds must be provided"
    assert probabilities is not None, "probabilities must be provided"
    assert transforms is not None, "transforms must be provided"

    assert probabilities.ndim == 1, f"got {probabilities.ndim}"
    assert thresholds.ndim == 1, f"got {thresholds.ndim}"
    ntransforms = len(transforms)
    assert ntransforms == thresholds.shape[0], f"got {ntransforms} transforms and {thresholds.shape[0]} thresholds"

    batch0 = batches[0]
    assert batch0.shape[0] == probabilities.shape[0], f"got {batch0.shape[0], probabilities.shape[0]}"
    
    # probabilities are between 0 and 1
    assert (0 <= probabilities < 1).all(), f"got probabilities {probabilities}, they must be between 0 and 1"
    assert (0 <= thresholds < 1).all(), f"got thresholds {thresholds}, they must be between 0 and 1"
    
    # the thresholds are sorted and cannot be the same (< instead of <=)
    assert (thresholds[:-1] < thresholds[1:]).all(), f"got thresholds {thresholds}, they must be sorted"
    
    # the reshapes + ">" will make a 2d table with true/false 
    # where "true" means the probability at line i is higher than the threshold at column j
    # so the sum(axis=-1) is counting how many thresholds the proba is higher than
    # the -1 makes it a 0-starting index
    selected_ops_indices = (probabilities.reshape(-1, 1) > thresholds.reshape(1, -1)).sum(axis=-1) - 1
    
    # idx: ndarray
    # the idx is the index of the transform in self.transforms that will be applied to the
    # group of instances whose indices are in ndarray
    for idx, transf in enumerate(transforms):
        
        select = selected_ops_indices == idx
        
        # it is importante tha a MultiBatchTransformMixin gets a list of tensors and not one by one
        # because if it is a random transformation, it should make sure that the same transformation
        # is applied to each batch
        assert isinstance(transf, MultiBatchTransformMixin), f"transforms must be MultiBatchTransformMixin, got {transf} at index {idx}"
        transformed_baches_pieces = transf(*[batch[select] for batch in batches])
        
        for idx, transformed_batch_piece in enumerate(transformed_baches_pieces):
            batches[idx][select] = transformed_batch_piece       
        
    return batches


class MultiBatchdRandomChoice(NumpyRandomTransformMixin, MultiBatchTransformMixin, TransformsMixin):
    """
    Pick a transformation randomly picked from a list for each instance in the batch. 
    This transform does not support torchscript.
    Use probabilities intervals between 0 and 1 to decide which operation to use.
    Ex: if there are 4 operations, define 4 thresholds: 0, .25, .50, .75; 
        draw a sample from a uniform distribution before 0 and 1: .33 (e.g.)
        0. < .25 < .33* 
        .33 is bigger than 2 thresholds so the second operation (index=1)    
    """
    
    @NumpyRandomTransformMixin._init_generator
    @TransformsMixin._init_transforms
    def __init__(self, *args, **kwargs):
        self.ntransforms = len(self.transforms)
        self._thresholds = np.linspace(0, 1, self.ntransforms, endpoint=False)
    
    @MultiBatchTransformMixin._validate_before_and_after
    def __call__(self, *batches: List[Tensor]) -> List[Tensor]:
        # keep all the randomness separate from the transform logic
        batch0 = batches[0]
        batchsize = batch0.shape[0]
        probabilities = np.random.uniform(low=0, high=1, size=batchsize)
        return _apply_chosen_multibatch_transforms(
            *batches, 
            thresholds=self._thresholds, 
            probabilities=probabilities, 
            transforms=self.transforms
        )


# =============================================== COMPOSE TRANSFORMS ===============================================

class BatchCompose(BatchTransformMixin, TransformsMixin):
    """
    Like transforms.Compose, but it makes sure that all transforms are instances of BatchTransform.
    """
    
    @TransformsMixin._init_transforms
    def __init__(self) -> None:
        super().__init__()    

    @BatchTransformMixin._validate_before_and_after
    def __call__(self, batch: Tensor) -> Tensor:
        for t in self.transforms:
            batch = t(batch)
        return batch
    

class MultiBatchCompose(MultiBatchTransformMixin, TransformsMixin):
    """
    Like transforms.Compose, but applies all transformations to a multitude of batches, instead of just one.
    """
    
    @TransformsMixin._init_transforms
    def __init__(self):
        
        for idx, transform in enumerate(self.transforms):
            assert isinstance(transform, BatchTransformMixin), f"transform {transform.__class__.__name__} at index {idx} must be a {BatchTransformMixin.__name__}"
            
            if isinstance(transform, NumpyRandomTransformMixin):
                transform._init_generator()
    
    @MultiBatchTransformMixin._validate_before_and_after
    def __call__(self, *batches: List[Tensor]) -> List[Tensor]:
        
        for t in self.transforms:
            batches = t(*batches)
            
        return batches

    
# =============================================== MISC BATCH TRANSFORMS ===============================================


class BatchRandomCrop(BatchTransformMixin, NumpyRandomTransformMixin, torch.nn.Module):
    """
    Padding is not supported!
    """
    
    @NumpyRandomTransformMixin._init_generator
    def __init__(self, size):
        super().__init__()
        self.size: Tuple[int, int] = tuple(T._setup_size(
            size, 
            error_msg="Please provide only two dimensions (h, w) for size."
        ))
        
    @staticmethod
    def get_params(batch: Tensor, output_size: Tuple[int, int], generator: np.random.Generator) -> Tuple[Tensor, Tensor]:
        """
        Get a batch of random crop parameters: [i, j]

        Args:
            batch (batch): Images to be cropped.
            output_size (tuple): Expected output size of the crop.

        Returns:
            (Tensor,Tensor): (i, j), each \in int64^B, where B is the batch size.
            i is for the height direction, j is for the width direction.
        """
        batchsize, _, h, w = batch.shape
        th, tw = output_size

        if h + 1 < th or w + 1 < tw:
            raise ValueError(
                "Required crop size {} is larger then input image size {}".format((th, tw), (h, w))
            )

        if w == tw and h == th:
            return torch.tensor([[0, 0, h, w]]).repeat((batchsize, 1))

        i = torch.tensor(generator.integers(low=0, high=h - th + 1, size=(batchsize,), dtype=np.int64))
        j = torch.tensor(generator.integers(low=0, high=w - tw + 1, size=(batchsize,), dtype=np.int64))
        
        return i, j
    
    @staticmethod
    def crop(batch: Tensor, top: Tensor, left: Tensor, output_size: Tuple[int, int], ) -> Tensor:
        assert batch.shape[0] == top.shape[0] == left.shape[0], f"batch, top, left, must have the same batch size, but got {batch.shape[0]}, {top.shape[0]}, {left.shape[0]}"
       
        assert (left >= 0).all(), f"left must be >= 0, but got {left}"
        assert (top >= 0).all(), f"top must be >= 0, but got {top}"
        
        batchsize, nchannels, im_height, im_width = batch.shape
        height, width = output_size
        
        right = left + width
        bottom = top + height
        
        assert (right <= im_width).all(), f"right must be <= image width, but got {right}"
        assert (bottom <= im_height).all(), f"bottom must be <= image height, but got {bottom}"
        
        height_crop = torch.arange(im_height).expand(batchsize, im_height)
        height_crop = torch.logical_and(
            top.view(-1, 1) <= height_crop, 
            height_crop < bottom.view(-1, 1),
        )
        batch = batch.transpose(0, 1)\
                [:, height_crop, :]\
                .view((nchannels, batchsize) + (height, im_width))
            
        width_crop = torch.arange(im_width).expand(batchsize, im_width)
        width_crop = torch.logical_and(
            left.view(-1, 1) <= width_crop, 
            width_crop < right.view(-1, 1),
        )

        batch = batch.transpose(-2, -1)\
                [:, width_crop, :]\
                .view((nchannels, batchsize) + (width, height))\
                .transpose(-2, -1)\
                .transpose(0, 1)
        
        return batch

    @BatchTransformMixin._validate_before_and_after
    def forward(self, batch: Tensor) -> Tensor:
        i, j = self.get_params(batch, self.size, self.generator)
        return self.crop(batch, i, j, self.size)         
    


class BatchRandomCrop_ForLoop(BatchTransformMixin, NumpyRandomTransformMixin, torch.nn.Module):
    """
    Padding is not supported!
    """
    
    @NumpyRandomTransformMixin._init_generator
    def __init__(self, size):
        super().__init__()
        self.size: Tuple[int, int] = tuple(T._setup_size(
            size, 
            error_msg="Please provide only two dimensions (h, w) for size."
        ))
        
    @staticmethod
    def get_params(batch: Tensor, output_size: Tuple[int, int], generator: np.random.Generator) -> Tuple[Tensor, Tensor]:
        """
        Get a batch of random crop parameters: [i, j]

        Args:
            batch (batch): Images to be cropped.
            output_size (tuple): Expected output size of the crop.

        Returns:
            (Tensor,Tensor): (i, j), each \in int64^B, where B is the batch size.
            i is for the height direction, j is for the width direction.
        """
        batchsize, _, h, w = batch.shape
        th, tw = output_size

        if h + 1 < th or w + 1 < tw:
            raise ValueError(
                "Required crop size {} is larger then input image size {}".format((th, tw), (h, w))
            )

        if w == tw and h == th:
            return torch.tensor([[0, 0, h, w]]).repeat((batchsize, 1))

        i = torch.tensor(generator.integers(low=0, high=h - th + 1, size=(batchsize,), dtype=np.int64))
        j = torch.tensor(generator.integers(low=0, high=w - tw + 1, size=(batchsize,), dtype=np.int64))
        
        return i, j
    
    @staticmethod
    def crop(batch: Tensor, top: Tensor, left: Tensor, output_size: Tuple[int, int], ) -> Tensor:
        assert batch.shape[0] == top.shape[0] == left.shape[0], f"batch, top, left, must have the same batch size, but got {batch.shape[0]}, {top.shape[0]}, {left.shape[0]}"
       
        assert (left >= 0).all(), f"left must be >= 0, but got {left}"
        assert (top >= 0).all(), f"top must be >= 0, but got {top}"
        
        batchsize, nchannels, im_height, im_width = batch.shape
        height, width = output_size
        
        right = left + width
        bottom = top + height
        
        assert (right <= im_width).all(), f"right must be <= image width, but got {right}"
        assert (bottom <= im_height).all(), f"bottom must be <= image height, but got {bottom}"
        
        crop_batch = torch.empty((batchsize, nchannels, height, width))
        for idx in range(batchsize):
            crop_batch[idx] = batch[idx, :, top[idx]:bottom[idx], left[idx]:right[idx]]
                
        return crop_batch

    @BatchTransformMixin._validate_before_and_after
    def forward(self, batch: Tensor) -> Tensor:
        i, j = self.get_params(batch, self.size, self.generator)
        return self.crop(batch, i, j, self.size)    

from common_dev01 import create_numpy_random_generator

random_batch = torch.randn((128, 3, 260, 260), device="cuda")
crop_size = (224, 224)
generator = create_numpy_random_generator(0)
     
batch_random_crop = BatchRandomCrop(crop_size, generator=generator)
batch_random_crop_foorloop = BatchRandomCrop_ForLoop(crop_size, generator=generator)

# In[]           
%timeit _ = batch_random_crop(random_batch)
# In[]        
%timeit _ = batch_random_crop_foorloop(random_batch)
# In[]        
        
# =============================================== DATALOADER PREVIEW ===============================================


def generate_dataloader_images(dataloader: DataLoader, nimages_perclass=20) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Generates a preview of the dataset, i.e. it generates an image of some randomly chosen outputs
    of the dataloader, including ground-truth maps.
    The data samples already have been augmented by the preprocessing pipeline.
    This method is useful to have an overview of how the preprocessed samples look like and especially
    to have an early look at the artificial anomalies.
    :param nimages_perclass: how many samples are shown per class, i.e. for anomalies and nominal samples each
    :return: four Tensors of images of shape (n x c x h x w): ((normal_imgs, normal_gtmaps), (anomalous_imgs, anomalous_gtmaps)) 
    """
    print('Generating images...')
    
    imgs, y, gtmaps = torch.FloatTensor(), torch.LongTensor(), torch.FloatTensor()
    
    for imgb, yb, gtmapb in dataloader:
        
        imgs, y, gtmaps = torch.cat([imgs, imgb]), torch.cat([y, yb]), torch.cat([gtmaps, gtmapb])
        
        # the number of anomalous in a batch can be random, so we have to keep iterating and checking 
        if (y == NOMINAL_TARGET).sum() >= nimages_perclass and (y == ANOMALY_TARGET).sum() >= nimages_perclass:
            break
        
    n_unique_values_in_gts = len(set(gtmaps.reshape(-1).tolist()))
    assert n_unique_values_in_gts <= 2, 'training process assumes zero-one gtmaps'
    
    effective_nimages_perclass = min(min(Counter(y.tolist()).values()), nimages_perclass)
    if effective_nimages_perclass < nimages_perclass:
        print(f"could not find {nimages_perclass} on each class, only generated {effective_nimages_perclass} of each")
    
    ret = (
        imgs[y == NOMINAL_TARGET][:effective_nimages_perclass], 
        gtmaps[y == NOMINAL_TARGET][:effective_nimages_perclass],
        imgs[y == ANOMALY_TARGET][:effective_nimages_perclass], 
        gtmaps[y == ANOMALY_TARGET][:effective_nimages_perclass],
    )
    
    print('Images generated.')
    return ret
    

def generate_dataloader_preview_multiple_fig(
    normal_imgs: Tensor, 
    normal_gtmaps: Tensor, 
    anomalous_imgs: Tensor,
    anomalous_gtmaps: Tensor,
) -> List[Figure]:
    """
    normal_imgs, anomalous_imgs: tensors of shape (n x 3 x h x w)
    normal_gtmaps,anomalous_gtmaps: tensors of shape (n x 1 x h x w)
    
    example code to use this function:
    
    ```
    ```
    """
    print('Generating dataset preview...')

    assert normal_imgs.shape[1] == 3, f"normal imgs: expected 3 channels, got {normal_imgs[1]}"
    assert normal_gtmaps.shape[1] == 1, f"normal gtmaps: expected 1 channel, got {normal_gtmaps[1]}"
    
    assert anomalous_imgs.shape == normal_imgs.shape, f"images have different shapes: anomalous:{anomalous_imgs.shape} vs normal:{normal_imgs.shape}"
    assert anomalous_gtmaps.shape == normal_gtmaps.shape, f"gtmaps have different shapes: anomalous:{anomalous_gtmaps.shape} vs normal:{normal_gtmaps.shape}"

    assert anomalous_imgs.shape[0] == anomalous_gtmaps.shape[0], f"anomalous images and gtmaps have different number of samples: anomalous:{anomalous_imgs.shape[0]} vs normal:{anomalous_gtmaps.shape[0]}"
    assert anomalous_imgs.shape[2:] == anomalous_gtmaps.shape[2:], f"anomalous images and gtmaps have different shapes: anomalous:{anomalous_imgs.shape} vs anomalous:{anomalous_gtmaps.shape}"

    assert normal_imgs.shape[0] == normal_gtmaps.shape[0], f"normal images and gtmaps have different number of samples: imgs:{normal_imgs.shape[0]} vs gtmaps:{normal_gtmaps.shape[0]}"
    assert normal_imgs.shape[2:] == normal_gtmaps.shape[2:], f"normal images and gtmaps have different shapes: imgs:{normal_imgs.shape} vs gtmaps:{normal_gtmaps.shape}"

    _, _, height, width = normal_imgs.shape
    
    def do(imgs: Tensor, gtmaps: Tensor, label_prefix: str) -> List[Figure]:
        """([n, 3, h, w], [n, 1, h, w]) -> n plt figures"""
        # 2 accounts for: img, gtmap
        nonlocal width, height
        figsize = (width, 2 * height)
        prevs = [
            # dim 1 = height ==> vertical concatenation
            # the repeat(3) is there to make the gtmap "RGB" while keeping it gray 
            torch.cat([img, gtmap.repeat(3, 1, 1)], dim=1)
            for img, gtmap in zip(imgs, gtmaps)
        ]
        figs = []
        for idx, prev in enumerate(prevs):
            
            # this peculiar way of creating a figure is because of the way matplotlib works
            # so i can get a png at the same size as the original image in terms of pixels
            # src: https://stackoverflow.com/a/13714915/9582881
            fig = plt.figure(frameon=False)
            fig.set_size_inches(*figsize)
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.set_axis_off()
            fig.add_axes(ax)
 
            ax.imshow(np.transpose(prev.numpy(), (1, 2, 0)), aspect=1)   
            fig.label = f"{label_prefix}_{idx}"

            figs.append(fig)         
 
        return figs
    
    normal_figs = do(normal_imgs, normal_gtmaps, "normal")
    anomalous_figs = do(anomalous_imgs, anomalous_gtmaps, "anomalous")
    
    print('Dataset preview generated.')
    
    return normal_figs, anomalous_figs


def generate_dataloader_preview_single_fig(
    normal_imgs: Tensor, 
    normal_gtmaps: Tensor, 
    anomalous_imgs: Tensor,
    anomalous_gtmaps: Tensor,
) -> matplotlib.figure.Figure:
    """
    normal_imgs, anomalous_imgs: tensors of shape (n x 3 x h x w)
    normal_gtmaps,anomalous_gtmaps: tensors of shape (n x 1 x h x w)
    
    example code to use this function:
    
    ```
    ```
    
    """
    print('Generating dataset preview...')

    assert normal_imgs.shape[1] == 3, f"normal imgs: expected 3 channels, got {normal_imgs[1]}"
    assert normal_gtmaps.shape[1] == 1, f"normal gtmaps: expected 1 channel, got {normal_gtmaps[1]}"
    
    assert anomalous_imgs.shape == normal_imgs.shape, f"images have different shapes: anomalous:{anomalous_imgs.shape} vs normal:{normal_imgs.shape}"
    assert anomalous_gtmaps.shape == normal_gtmaps.shape, f"gtmaps have different shapes: anomalous:{anomalous_gtmaps.shape} vs normal:{normal_gtmaps.shape}"

    assert anomalous_imgs.shape[0] == anomalous_gtmaps.shape[0], f"anomalous images and gtmaps have different number of samples: anomalous:{anomalous_imgs.shape[0]} vs normal:{anomalous_gtmaps.shape[0]}"
    assert anomalous_imgs.shape[2:] == anomalous_gtmaps.shape[2:], f"anomalous images and gtmaps have different shapes: anomalous:{anomalous_imgs.shape} vs anomalous:{anomalous_gtmaps.shape}"

    assert normal_imgs.shape[0] == normal_gtmaps.shape[0], f"normal images and gtmaps have different number of samples: imgs:{normal_imgs.shape[0]} vs gtmaps:{normal_gtmaps.shape[0]}"
    assert normal_imgs.shape[2:] == normal_gtmaps.shape[2:], f"normal images and gtmaps have different shapes: imgs:{normal_imgs.shape} vs gtmaps:{normal_gtmaps.shape}"

    nimages, _, height, width = normal_imgs.shape
    
    def do(imgs: Tensor, gtmaps: Tensor) -> List[Figure]:
        """([n, 3, h, w], [n, 1, h, w]) -> n plt figures"""
        
        # concatenates img/gtmap vertically
        prevs = [
            # dim 1 = height ==> vertical concatenation
            torch.cat([img, gtmap.repeat(3, 1, 1)], dim=1)
            # the repeat(3, 1, 1) is there to make the gtmap "RGB" while keeping it gray 
            for img, gtmap in zip(imgs, gtmaps)
        ]
        # concatenate the many prevs horizontally
        return torch.cat(prevs, dim=2)
    
    normal_prevs = do(normal_imgs, normal_gtmaps)
    anomalous_prevs = do(anomalous_imgs, anomalous_gtmaps)
    
    # concatenate normal/anomalous previews vertically
    prev = torch.cat([normal_prevs, anomalous_prevs], dim=1)
    
    # 4 accounts for: normal img, normal gtmap, anomalous img, anomalous gtmap
    figsize = (width, height) = (nimages * width, 4 * height)
    
    # this peculiar way of creating a figure is because of the way matplotlib works
    # so i can get a png at the same size as the original image in terms of pixels
    # src: https://stackoverflow.com/a/13714915/9582881
    fig = plt.figure(frameon=False)
    fig.set_size_inches(*figsize)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    
    ax.imshow(np.transpose(prev.numpy(), (1, 2, 0)), aspect=1)   
    fig.label = f"preview_{nimages:02d}_images"
    
    print('Dataset preview generated.')
    
    return fig


# about test: create option to use original gtmaps or not (resized gtmaps)
# todo dont forget to manage channel dim (dont repeat indices 3 times)


