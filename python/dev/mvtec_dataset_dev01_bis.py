import functools
import hashlib
import os
import tarfile
import tempfile
from itertools import cycle
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import numpy.random
import torch
import torchvision.transforms as transforms
from PIL import Image
from pytorch_lightning import LightningDataModule
from six.moves import urllib
from skimage.transform import rotate as im_rotate
from torch import Tensor
from torch.nn.functional import interpolate as torch_interpolate
from torch.utils.data import DataLoader  #, Subset
from torch.utils.data.dataset import Dataset
from torchvision.datasets import VisionDataset
from torchvision.datasets.imagenet import check_integrity
from tqdm import tqdm

from common_dev01_bis import (create_numpy_random_generator,
                          create_torch_random_generator)
from data_dev01_bis import (ANOMALY_TARGET, NOMINAL_TARGET, BatchCompose,
                        BatchGaussianNoise, BatchLocalContrastNormalization,
                        BatchRandomChoice, BatchRandomCrop, BatchRandomFlip, LightningDataset, LoopBatchRandomAffine, MultiBatchCompose,
                        MultiBatchdRandomChoice, MultiBatchTransformMixin,
                        RandomTransformMixin, generate_dataloader_images,
                        generate_dataloader_preview_single_fig,
                        make_multibatch, make_multibatch_use_same_random_state)

DATASET_NAME = "mvtec"

CLASSES_LABELS = (
    'bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather',
    'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor',
    'wood', 'zipper'
)

CLASSES_FULLQUALIFIED = tuple(
    f"{DATASET_NAME}_{idx:02d}_{label}" 
    for idx, label in enumerate(CLASSES_LABELS)
)

CLASS_TYPE_OBJECT = 'object'
CLASS_TYPE_TEXTURE = "texture"
CLASSES_TYPES = (
    CLASS_TYPE_OBJECT, CLASS_TYPE_OBJECT, CLASS_TYPE_OBJECT, CLASS_TYPE_TEXTURE, CLASS_TYPE_TEXTURE, CLASS_TYPE_OBJECT, CLASS_TYPE_TEXTURE,
    CLASS_TYPE_OBJECT, CLASS_TYPE_OBJECT, CLASS_TYPE_OBJECT, CLASS_TYPE_TEXTURE, CLASS_TYPE_OBJECT, CLASS_TYPE_OBJECT,
    CLASS_TYPE_TEXTURE, CLASS_TYPE_OBJECT
)


NCLASSES = len(CLASSES_LABELS)

NORMAL_LABEL = 'good'
NORMAL_LABEL_IDX = 0

SPLIT_TRAIN = "train"
SPLIT_TEST = "test"
SPLITS = (SPLIT_TRAIN, SPLIT_TEST)

SUPERVISE_MODE_REAL_ANOMALY = 'real-anomaly'
SUPERVISE_MODE_SYNTHETIC_ANOMALY_CONFETTI = 'synthetic-anomaly-confetti'
SUPERVISE_MODES = (SUPERVISE_MODE_REAL_ANOMALY, SUPERVISE_MODE_SYNTHETIC_ANOMALY_CONFETTI)
SUPERVISE_MODES_DOC = {
    SUPERVISE_MODE_SYNTHETIC_ANOMALY_CONFETTI: "normal data is used and a synthetic anomaly is added using confetti noise",
    SUPERVISE_MODE_REAL_ANOMALY: "real anomalous cases are used for supervision",
}

PREPROCESSING_LCNAUG1 = 'lcnaug1'
PREPROCESSING_LCNAUG2 = 'lcnaug2'
PREPROCESSING_CHOICES = (PREPROCESSING_LCNAUG1, PREPROCESSING_LCNAUG2)

DATAMODULE_PREPROCESS_MOMENT_BEFORE_BATCH_TRANSFER = "preprocess-before-batch-transfer"
DATAMODULE_PREPROCESS_MOMENT_AFTER_BATCH_TRANSFER = "preprocess-after-batch-transfer"
DATAMODULE_PREPROCESS_MOMENT_CHOICES = (DATAMODULE_PREPROCESS_MOMENT_BEFORE_BATCH_TRANSFER, DATAMODULE_PREPROCESS_MOMENT_AFTER_BATCH_TRANSFER)

TARGZ_DOWNLOAD_URL = "ftp://guest:GU%2E205dldo@ftp.softronics.ch/mvtec_anomaly_detection/mvtec_anomaly_detection.tar.xz"
TARGZ_FNAME = 'mvtec_anomaly_detection.tar.xz'

BASE_FOLDER = 'mvtec'


def confetti_noise(
    shape: Tuple[int, int, int, int], 
    probability_threshold: float = 0.01,
    blobshaperange: Tuple[Tuple[int, int], Tuple[int, int]] = ((3, 3), (5, 5)),
    fillval: int = 255, 
    backval: int = 0, 
    ensureblob: bool = True, 
    awgn: float = 0.0,
    clamp: bool = False, 
    onlysquared: bool = True, 
    rotation: int = 0,
    colorrange: Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]] = None, 
    numpy_generator: numpy.random.Generator = None,  # it should never be none, but i put it like this to not change the signature
    torch_generator: torch.Generator = None,  # it should never be none, but i put it like this to not change the signature
    dtype=torch.float32,
) -> Tensor:
    """
    Generates "confetti" noise, as seen in the paper.
    The noise is based on sampling randomly many rectangles (in the following called blobs) at random positions.
    Additionally, all blobs are of random size (within some range), of random rotation, and of random color.
    The color is randomly chosen per blob, thus consistent within one blob.
    :param size: size of the overall noise image(s), should be (n x h x w) or (n x c x h x w), i.e.
        number of samples, channels, height, width. Blobs are grayscaled for (n x h x w) or c == 1.
    :param p: the probability of inserting a blob per pixel.
        The average number of blobs in the image is p * h * w.
    :param blobshaperange: limits the random size of the blobs. For ((h0, h1), (w0, w1)), all blobs' width
        is ensured to be in {w0, ..., w1}, and height to be in {h0, ..., h1}.
    :param fillval: if the color is not randomly chosen (see colored parameter), this sets the color of all blobs.
        This is also the maximum value used for clamping (see clamp parameter). Can be negative.
    :param backval: the background pixel value, i.e. the color of pixels in the noise image that are not part
         of a blob. Also used for clamping.
    :param ensureblob: whether to ensure that there is at least one blob per noise image.
    :param awgn: amount of additive white gaussian noise added to all blobs.
    :param clamp: whether to clamp all noise image to the pixel value range (backval, fillval).
    :param onlysquared: whether to restrict the blobs to be squares only.
    :param rotation: the maximum amount of rotation (in degrees)
    :param colorrange: the range of possible color values for each blob and channel.
        Defaults to None, where the blobs are not colored, but instead parameter fillval is used.
        First value can be negative.
    :return: torch tensor containing n noise images. Either (n x c x h x w) or (n x h x w), depending on size.
    """
    assert numpy_generator is not None, f"generator must not be None"
    assert torch_generator is not None, f"generator must not be None"
    
    assert isinstance(shape, tuple), f"shape must be a tuple, got {type(shape)}"
    assert len(shape) == 4, f'size must be n x c x h x w, but is {shape}'
    batchsize, nchannels, height, width = shape

    assert len(blobshaperange) == 2, f"blobshaperange must be a tuple of two tuples of two ints, but is {blobshaperange}"
    (heightrange, widthrange) = blobshaperange
    
    assert isinstance(heightrange, tuple) and isinstance(widthrange, tuple), f"blobshaperange must be a tuple of two tuples of two ints, but is {blobshaperange}"
    assert len(heightrange) == 2, f"heightrange must be (h0, h1), but is {heightrange}"
    assert len(widthrange) == 2, f"widthrange must be (w0, w1), but is {widthrange}"
    (hmin, hmax) = heightrange
    (wmin, wmax) = widthrange
    
    assert 0 < hmin <= hmax < height, f"heightrange must be (h0, h1) with 0 < h0 <= h1 < h, but is {heightrange}"
    assert 0 < wmin <= wmax < width, f"widthrange must be (w0, w1) with 0 < w0 <= w1 < w, but is {widthrange}"
    
    if colorrange is not None:
        assert nchannels == 3, f"colorrange can only be used with 3 channels, but is {colorrange} and {nchannels}"
        assert len(colorrange) == 2, f"colorrange must be a tuple of two tuples of three ints, but is {colorrange}"
    
    mask_shape = (batchsize, 1, height, width)
    
    # mask[i, j, k] == 1 for center of blob
    mask = (torch.rand(size=mask_shape, generator=torch_generator, dtype=dtype) < probability_threshold)
    
    nblobs_persample = mask.view(batchsize, -1).sum(1)

    while ensureblob and (nblobs_persample.min() == 0):
        # nonzero returns a 2-D tensor where each row is the index for a nonzero value.
        idx_zero_blob_centers = (nblobs_persample == 0).nonzero().squeeze()
        n_left = idx_zero_blob_centers.size(0) if len(idx_zero_blob_centers.shape) > 0 else 1
        rand_shape = (n_left, height, width)
        mask[idx_zero_blob_centers] = (torch.rand(rand_shape, dtype=dtype) < probability_threshold).unsqueeze(1)
        nblobs_persample = mask.view(batchsize, -1).sum(1)
    
    res = torch.full(size=shape, fill_value=backval, dtype=dtype)
    
    # [(idn, idz, idy, idx), ...] = indices of blob centers \in int^[batchsize, 4]
    # idz should always be 0 since we only have one channel
    blob_centers = mask.nonzero()  
    
    # if there are no blobs at all
    if blob_centers.reshape(-1).size(0) == 0:
        return torch.zeros(size=shape, dtype=dtype)

    all_shapes = [
        (x, y) 
        for x in range(wmin, wmax + 1)
        for y in range(hmin, hmax + 1) 
        if not onlysquared or x == y
    ]
    
    nshapes = len(all_shapes)
    ncenters = blob_centers.size(0)
    picked_shapes = torch.randint(size=(ncenters,), low=0, high=nshapes, dtype=torch.int64, generator=torch_generator)
    nidx = []
    blob_colors = []
    for shape_idx, blobshape in enumerate(all_shapes):
        
        npicked = (picked_shapes == shape_idx).sum()
        if npicked < 1:
            continue
        
        bhs = range(-(blobshape[0] // 2) if blobshape[0] % 2 != 0 else -(blobshape[0] // 2) + 1, blobshape[0] // 2 + 1)
        bws = range(-(blobshape[1] // 2) if blobshape[1] % 2 != 0 else -(blobshape[1] // 2) + 1, blobshape[1] // 2 + 1)
        extends = torch.stack([
            torch.zeros(len(bhs) * len(bws)).long(),
            torch.zeros(len(bhs) * len(bws)).long(),
            torch.arange(bhs.start, bhs.stop).repeat(len(bws)),
            torch.arange(bws.start, bws.stop).unsqueeze(1).repeat(1, len(bhs)).reshape(-1),
        ]).transpose(0, 1)
        # nid \in int^[number_blobs_of_this_size, blob_height * blob_width, 4]
        # and "4" to the indexes in the image being generated where there should be some value
        # i.e. [instance index, channel index, height index, width index]
        # blob_height * blob_width is the number of pixels in each blobs
        # number_blobs_of_this_size  ...
        nid = blob_centers[picked_shapes == shape_idx].unsqueeze(1) + extends.unsqueeze(0)
        (_, npixels_perblob, _) = nid.shape
    
        # merge the axis `subbatchsize` and `npixels_perblob`
        # "4" is the number of axis in shape
        # result: list of all indices that the blobs cove in the current *subbatch*
        nid = nid.reshape(-1, 4) 
        
        # make sure all the index values are valid
        nid = torch.max(torch.min(nid, torch.tensor(shape) - 1), torch.tensor([0, 0, 0, 0]))
        nidx.append(nid)
        
        # append the respective color of these blobs
        if colorrange is not None:
            colormin, colormax = colorrange
            # old = torch.randint(low=colormin, high=colormax, size=(nchannels,))[:, None].repeat(1, npixels_filled.size(0)).int()
            blobcolor = torch.randint(
                low=colormin, 
                high=colormax, 
                size=(npicked, nchannels,),
                dtype=dtype, 
                generator=torch_generator,
            ).repeat(npixels_perblob, 1)
            blob_colors.append(blobcolor)
        
    # all pixel indices that blobs cover, not only center indices
    nidx = torch.cat([nid.reshape(-1, 4) for nid in nidx]) 
    nblob_pixels = nidx.shape[0]
    
    # this is really necessary because these have 
    # to be in the format [4, n_pixels]
    # and it doesnt work with tensors for indexing 
    # so this numpy() is necessary, and it will be run 
    # before the data is moved to the GPU so it doesnt matter
    nidx = nidx.transpose(0, 1).numpy()
    
    def torchrandn(size_):
        return torch.randn(size=size_, dtype=dtype, generator=torch_generator)
    
    if colorrange is not None:
        # this should be the same legth as nidx, containing each pixel's color 
        blob_colors = torch.cat(blob_colors, dim=0).transpose(0, 1)
        
        # they are different type but the indexing will work (check the shape)
        if awgn == 0:
            gnoise = (0, 0, 0)

        else:
            gnoise = awgn * torchrandn(size_=(3, nblob_pixels,))
        
        for channel_idx in range(nchannels):
            pixel_select = nidx + np.array((0, channel_idx, 0, 0))[:, None]
            res[pixel_select] = blob_colors[channel_idx] + gnoise[channel_idx] 
            
    else:
        if awgn == 0:
            gnoise = 0
        else:
            gnoise = awgn * torchrandn(size_=(nblob_pixels,), )
            
        res[nidx] = torch.full(size=(nblob_pixels,), fill_value=fillval, dtype=dtype,) + gnoise
        # res = res.repeat(1, nchannels, 1, 1)
        
    if clamp:
        res = res.clamp(backval, fillval) if backval < fillval else res.clamp(fillval, backval)
        
    mask = mask[:, 0, :, :]
    
    if rotation > 0:
        
        def ceil(x: float):
            return int(np.ceil(x))
        
        def floor(x: float):
            return int(np.floor(x))
        
        # [b, c, h, w] -> [b, w, h, c] -> [b, h, w, c]
        res = res.transpose(1, 3).transpose(1, 2)
        
        for picked_shape_idx, blob_center in zip(picked_shapes, mask.nonzero()):
            blobh, blobw = all_shapes[picked_shape_idx]
            rot = numpy_generator.uniform(-rotation, rotation)
            nblobs = blob_center[0]
            dims = (
                nblobs,
                slice(max(blob_center[1] - floor(0.75 * blobh), 0), min(blob_center[1] + ceil(0.75 * blobh), res.size(1) - 1)),
                slice(max(blob_center[2] - floor(0.75 * blobw), 0), min(blob_center[2] + ceil(0.75 * blobw), res.size(2) - 1)),
                ...
            )
            res[dims] = torch.from_numpy(
                im_rotate(
                    res[dims].float(), 
                    rot, 
                    order=0, 
                    cval=0, 
                    center=(blob_center[1]-dims[1].start, blob_center[2]-dims[2].start),
                    clip=False
                ),
            ).float()
        # [b, c, h, w] <- [b, w, h, c] <- [b, h, w, c]
        res = res.transpose(1, 2).transpose(1, 3)
    return res


def merge_image_and_synthetic_noise(
        img: Tensor, 
        gt: Tensor, 
        generated_noise: Tensor, 
        invert_threshold: float = 0.00025,
    ):
        assert (img.dim() == 4 or img.dim() == 3) and generated_noise.shape == img.shape
        anom = img.clone()

        # invert noise if difference of malformed and original is less than threshold and inverted difference is higher
        diff = ((anom.int() + generated_noise).clamp(0, 255) - anom.int())
        diff = diff.reshape(anom.size(0), -1).sum(1).float().div(np.prod(anom.shape)).abs()
        
        diffi = ((anom.int() - generated_noise).clamp(0, 255) - anom.int())
        diffi = diffi.reshape(anom.size(0), -1).sum(1).float().div(np.prod(anom.shape)).abs()
        
        inv = [i for i, (d, di) in enumerate(zip(diff, diffi)) if d < invert_threshold and di > d]
        generated_noise[inv] = -generated_noise[inv]

        anom = (anom.int() + generated_noise).clamp(0, 255).byte()

        gt = (img != anom).max(1)[0].clone().float() 
        gt = gt.unsqueeze(1)  # value 1 will be put to anom_label in mvtec_bases get_item

        # anom = anom.float() / 255.0

        return anom, gt
    

class BatchOnlineInstanceReplacer(RandomTransformMixin, MultiBatchTransformMixin):
    
    invert_threshold = 0.025

    @RandomTransformMixin._init_generator
    @RandomTransformMixin._init_torch_generator
    def __init__(self, supervise_mode: str,p: float = 0.5, inplace=False):
        """
        This class is used as a Transform parameter for torchvision datasets.
        During training it randomly replaces a sample of the dataset retrieved via the get_item method
        by an anomaly (artificial or not).
        :param ds: some AD dataset for which the OnlineSupervisor is used.
        :param supervise_mode: the type of artificial anomalies to be generated during training.
        :param real_anomaly_limit: the number of different Outlier Exposure samples used in case of outlier exposure based noise.
        :param p: the chance to replace a sample from the original dataset during training.
        :param real_anomaly_dataloader: a dataloader for the real anomalies to be used in case of outlier exposure based noise (it will be ciclycally iterated).
        """
        assert supervise_mode in SUPERVISE_MODES, f"{supervise_mode} not in {SUPERVISE_MODES}"
        self.supervise_mode = supervise_mode
        self.p = p
        self._real_anomaly_dataloader = None
        
        assert not inplace, "please be careful to use this"
        self.inplace = inplace
    
    @property
    def real_anomaly_dataloader(self) -> torch.utils.data.DataLoader:
        assert self.supervise_mode == SUPERVISE_MODE_REAL_ANOMALY, f"real_anomaly_dataloader (@property) should only be called on supervise mode {SUPERVISE_MODE_REAL_ANOMALY}"
        assert self._real_anomaly_dataloader is not None, '_real_anomaly_dataloader is required for real anomaly mode, please set _real_anomaly_dataloader before getting it'
        return self._real_anomaly_dataloader_cycle
    
    @real_anomaly_dataloader.setter
    def real_anomaly_dataloader(self, dataloader: torch.utils.data.DataLoader):
        assert self.supervise_mode == SUPERVISE_MODE_REAL_ANOMALY, f"real_anomaly_dataloader should only be set on supervise mode {SUPERVISE_MODE_REAL_ANOMALY}"
        self._real_anomaly_dataloader = dataloader
        self._real_anomaly_dataloader_cycle = cycle(self._real_anomaly_dataloader)
    
    @MultiBatchTransformMixin._validate_consistent_batchsize
    @MultiBatchTransformMixin._validate_ndims((4, 1, 4))
    @MultiBatchTransformMixin._validate_before_and_after_is_same()
    def __call__(self, imgs: Tensor, labels: Tensor, gtmaps: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Based on the probability defined in __init__, replaces (img, gt, target) with an artificial anomaly.
        :param img: some torch tensor image
        :param gt: some ground-truth map (can be None)
        :param target: some label
        :return: (img, gt, target)
        """
        batchsize = imgs.shape[0]
        probabilities = self.generator.random(size=(batchsize,))
        replaced_indices = probabilities < self.p
        
        if replaced_indices.sum() == 0:
            return imgs, labels, gtmaps
        
        if self.inplace:
            imgs[replaced_indices], labels[replaced_indices], gtmaps[replaced_indices] = self._replace(imgs[replaced_indices], labels[replaced_indices], gtmaps[replaced_indices])
            return imgs, labels, gtmaps

        replacer_imgs, replacer_labels, replacer_gtmaps = self._replace(imgs[replaced_indices], labels[replaced_indices], gtmaps[replaced_indices]) 
      
        new_imgs = imgs.clone()
        new_labels = labels.clone()
        new_gtmaps = gtmaps.clone()        
            
        new_imgs[replaced_indices] = replacer_imgs
        new_labels[replaced_indices] = replacer_labels
        new_gtmaps[replaced_indices] = replacer_gtmaps
            
        return new_imgs, new_labels, new_gtmaps

    def _replace(self, imgs: Tensor, labels: Tensor, gtmaps: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
                
        if self.supervise_mode == SUPERVISE_MODE_SYNTHETIC_ANOMALY_CONFETTI:
            generated_noise_rgb = confetti_noise(
                imgs.shape, 
                0.000018, 
                ((8, 54), (8, 54)), 
                fillval=255, 
                clamp=False, 
                rotation=45, 
                colorrange=(-256, 0),
                numpy_generator=self.generator,
                torch_generator=self.torch_generator,
            )
            generated_noise = confetti_noise(
                imgs.shape, 
                0.000012, 
                ((8, 54), (8, 54)), 
                fillval=-255, 
                clamp=False, 
                awgn=0, 
                rotation=45,
                numpy_generator=self.generator,
                torch_generator=self.torch_generator,
            )
            generated_noise = generated_noise_rgb + generated_noise
            # generated_noise = smooth_noise(generated_noise, 25, 5, 1.0)
            
            imgs, gtmaps = merge_image_and_synthetic_noise(
                (imgs * 255).int(), 
                gtmaps.squeeze(), 
                generated_noise, 
                invert_threshold=self.invert_threshold,
            )
            imgs = imgs.float() / 255.0
            return_labels = torch.full(size=labels.shape, fill_value=ANOMALY_TARGET, dtype=labels.dtype)
            return imgs, return_labels, gtmaps
            
        elif self.supervise_mode == SUPERVISE_MODE_REAL_ANOMALY:
            
            batchsize = imgs.shape[0]
            
            imgs_new: Tensor = torch.tensor([], dtype=imgs.dtype, device=imgs.device)
            labels_new: Tensor = torch.tensor([], dtype=labels.dtype, device=labels.device)
            gtmaps_new: Tensor = torch.tensor([], dtype=gtmaps.dtype, device=gtmaps.device)
            # gtmaps_new.tolist()
            
            for imgs_, labels_, gtmaps_ in self.real_anomaly_dataloader:
                
                assert imgs_.shape[0] == labels_.shape[0] == gtmaps_.shape[0], f"imgs.shape={imgs_.shape}, labels.shape={labels_.shape}, gtmaps.shape={gtmaps_.shape}"
                assert imgs_.shape[1:] == imgs.shape[1:], f"imgs_ and imgs should have the same instance shape (shape[1:]), but got {imgs_.shape} and {imgs.shape}"
                assert labels_.shape[1:] == labels.shape[1:], f"labels_ and labels should have the same instance shape (shape[1:]), but got {labels_.shape} and {labels.shape}"
                assert gtmaps_.shape[1:] == gtmaps.shape[1:], f"gtmaps_ and gtmaps should have the same instance shape (shape[1:]), but got {gtmaps_.shape} and {gtmaps.shape}"
                labels_unique_values = set(labels_.unique().tolist())
                assert labels_unique_values == {ANOMALY_TARGET}, f"labels_ should only contain {ANOMALY_TARGET}, but got {labels_unique_values}" 
                
                imgs_new = torch.cat((imgs_new, imgs_), dim=0)
                labels_new = torch.cat((labels_new, labels_), dim=0)
                gtmaps_new = torch.cat((gtmaps_new, gtmaps_), dim=0)
                
                if imgs_new.shape[0] >= batchsize:
                    break
            
            imgs = imgs_new[:batchsize]
            labels = labels_new[:batchsize]
            gtmaps = gtmaps_new[:batchsize]

            return imgs, labels, gtmaps
        
        else:
            raise NotImplementedError('Supervise mode {self.supervise_mode} unknown.')
        

class MvTec(VisionDataset, Dataset, LightningDataset):
    """ 
    Implemention of a torch style MVTec dataset.
    0: nominal
    1: anomalous
    order of transforms:
        all_transform
        img_and_gtmap_transform
        img_transform
    """

    def __init__(
        self, 
        root: str, 
        split: str, 
        normal_class: int, 
        shape: Tuple[int, int, int] = (240, 240),
        img_and_gtmap_transform: Callable = None, 
        img_transform: Callable = None, 
        all_transform: Callable = None,
    ):
        """
        0: nominal
        1: anomalous
        order of transforms:
            all_transform
            img_and_gtmap_transform
            img_transform
        Loads all data from the prepared torch tensors. If such torch tensors containg MVTec data are not found
        in the given root directory, instead downloads the raw data and prepares the tensors.
        They contain labels, images, and ground-truth maps for a fixed size, determined by the shape parameter.
        :param root: directory where the data is to be found.
        :param split: whether to use "split_train", "split_test", "split_test_anomaly_type_label" data.
            In the latter case the get_item method returns labels indexing the anomalous class rather than
            the object class. That is, instead of returning 0 for "bottle", it returns "1" for "large_broken".
        :param img_and_gtmap_transform: function that takes image and ground-truth map and transforms it somewhat.
            Useful to apply the same augmentation to image and ground-truth map (e.g. cropping), s.t.
            the ground-truth map still matches the image.
            ImgGt transform is the third transform that is applied.
        :param img_transform: function that takes image and transforms it somewhat.
            Transform is the last transform that is applied.
        :param all_transform: function that takes image, label, and ground-truth map and transforms it somewhat.
            All transform is the second transform that is applied.
        :param shape: the shape (c x h x w) the data should be resized to (images and ground-truth maps).
        :param normal_classes: all the classes that are considered nominal (usually just one).
        """
        
        super(MvTec, self).__init__(
            # args of the VisionDataset
            root=root, 
            transform=img_transform, 
            target_transform=None,
        )
        
        self.rootpath = Path(root).absolute()
        self.basepath = self.rootpath / BASE_FOLDER
        
        assert split in SPLITS, f"Split must be one of {SPLITS}"
        self.split = split
        
        # validate shape
        assert len(shape) == 2, "shape must be a tuple of length 3"
        width, height = shape
        assert width == height, f"shape: width={width} != height={height}"
        del width, height
        self.shape = shape
        
        # there are 15 classes in the MVTEC dataset´
        assert 0 <= normal_class <= 14, f"normal_class must be in range(15), found {normal_class}"
        self.normal_class = normal_class
        
        # i will make a copy of the attributes of the VisionDataset 
        # by changing the names to avoid confusion
        self.img_transform = self.transform
        # and these ones are manually implemented
        self.img_and_gtmap_transform = img_and_gtmap_transform
        self.all_transform = all_transform
    
        # these are properties that must be loaded with setup()      
        self._imgs = None
        self._labels = None
        self._gtmaps = None
        self._anomaly_labels = None
        self._anomaly_label_strings = None
        self._original_gtmaps = None
        
        # these will restraint the set of images to a subset
        # it's used for when picking images from the test set 
        # to use as anomalies during the training
        self._subset_indices = None
    
    # todo make subset-able feature a mixin
    @property
    def is_subset(self) -> bool:
        return self._subset_indices is not None
    
    @property
    def subset_indices(self) -> List[int]:
        assert self._subset_indices is not None, "subset_indices not set"
        return self._subset_indices
    
    @subset_indices.setter
    def subset_indices(self, indices: List[int]):
        assert self._subset_indices is None, "subset_indices already set"
        assert len(indices) <= len(self.imgs), f"subset_indices must be a subset of the full dataset, but got {len(indices)} and {len(self.imgs)}"
        assert all(0 <= i < len(self.imgs) for i in indices), f"subset_indices must be a subset of the full dataset (of size {len(self.imgs)}), but got {indices}"
        self._subset_indices = indices
                
    @property
    def normal_class_label(self):
        return CLASSES_LABELS[self.normal_class]
    
    @property
    def imgs(self):
        if self._imgs is None:
            raise RuntimeError("MVTec dataset not setup yet. Please call setup() first.")
        return self._imgs        

    @imgs.setter
    def imgs(self, value):
        if self._imgs is not None:
            raise RuntimeError("imgs is already set.")
        assert value is not None, f"imgs cannot be set to None"
        self._imgs = value

    @property
    def labels(self):
        if self._labels is None:
            raise RuntimeError("MVTec dataset not setup yet. Please call setup() first.")
        return self._labels        

    @labels.setter
    def labels(self, value):
        if self._labels is not None:
            raise RuntimeError("labels is already set.")
        assert value is not None, f"labels cannot be set to None"
        self._labels = value

    @property
    def gtmaps(self):
        if self._gtmaps is None:
            raise RuntimeError("MVTec dataset not setup yet. Please call setup() first.")
        return self._gtmaps    

    @gtmaps.setter
    def gtmaps(self, value):
        if self._gtmaps is not None:
            raise RuntimeError("gtmaps is already set.")
        assert value is not None, f"gtmaps cannot be set to None"
        self._gtmaps = value

    @property
    def anomaly_labels(self):
        if self._anomaly_labels is None:
            raise RuntimeError("MVTec dataset not setup yet. Please call setup() first.")
        return self._anomaly_labels        

    @anomaly_labels.setter
    def anomaly_labels(self, value):
        if self._anomaly_labels is not None:
            raise RuntimeError("anomaly_labels is already set.")
        assert value is not None, f"anomaly_labels cannot be set to None"
        self._anomaly_labels = value

    @property
    def anomaly_label_strings(self):
        if self._anomaly_label_strings is None:
            raise RuntimeError("MVTec dataset not setup yet. Please call setup() first.")
        return self._anomaly_label_strings        

    @anomaly_label_strings.setter
    def anomaly_label_strings(self, value):
        if self._anomaly_label_strings is not None:
            raise RuntimeError("anomaly_label_strings is already set.")
        assert value is not None, f"anomaly_label_strings cannot be set to None"
        self._anomaly_label_strings = value

    @property
    def original_gtmaps(self):
        if self._original_gtmaps is None:
            raise RuntimeError("MVTec dataset not setup yet. Please call setup() first.")
        return self._original_gtmaps               

    @original_gtmaps.setter
    def original_gtmaps(self, value):
        if self._original_gtmaps is not None:
            raise RuntimeError("original_gtmaps is already set.")
        assert value is not None, f"original_gtmaps cannot be set to None"
        self._original_gtmaps = value

    @property
    def tar_fpath(self):
        return self.basepath / TARGZ_FNAME
    
    @staticmethod
    def data_fname(cls: int, shape: Tuple[int, int, int]):
        return f"admvtec_class_{cls:02}_shape_w{shape[1]:04}xh{shape[0]:04}.pt"
    
    @property
    def data_fpath(self) -> Path:
        return self.basepath / MvTec.data_fname(self.normal_class, self.shape)
    
    @property
    def _data_md5_fpath(self) -> Path:
        return self.data_fpath.parent / (self.data_fpath.name + ".md5")
    
    @staticmethod
    def original_data_fname(cls: int) -> Path:
        return f"admvtec_class_{cls:02}_original.pt"
    
    @property
    def original_data_fpath(self) -> Path:
        return self.basepath / MvTec.original_data_fname(self.normal_class)
    
    @property
    def _original_data_md5_fpath(self) -> Path:
        return self.original_data_fpath.parent / (self.original_data_fpath.name + ".md5")
        
    def prepare_data(self):
        """make sure the neceassary files with data exist"""
        
        self.basepath.mkdir(parents=True, exist_ok=True)
        
        if check_integrity(
            str(self.data_fpath), 
            self._data_md5_fpath.read_text() 
            if self._data_md5_fpath.exists() else 
            None
        ):
            print(f'File `{self.data_fpath}` already prepared.')
            assert check_integrity(
                str(self.original_data_fpath), 
                self._original_data_md5_fpath.read_text()
                if self._original_data_md5_fpath.exists() else
                None
            ), \
                f"file {self.data_fpath} is already prepared but its original {self.original_data_fpath} seems to be missing or corrupted"
            return 
        
        print(f'File `{self.data_fpath}` will be precomputed.')
        
        if check_integrity(
            str(self.original_data_fpath), 
            (
                self._original_data_md5_fpath.read_text() 
                if self._original_data_md5_fpath.exists() else 
                None    
            )
        ):
            print(f'Original file `{self.original_data_fpath}` already prepared.')
            
        else:
            print(f'Original file `{self.original_data_fpath}` will be precomputed.')
            
            if self.tar_fpath.exists():
                 print(f".tar.gz file `{self.tar_fpath}` already downloaded")
            
            else:
                
                def download_targz(url, dir, filename=None):
                    """Download a file from a url and place it in dir.
                    Args:
                        url (str): URL to download file from
                        root (str): Directory to place downloaded file in
                        filename (str, optional): Name to save the file under. If None, use the basename of the URL
                    """

                    dir = os.path.expanduser(dir)
                    
                    if not filename:
                        filename = os.path.basename(url)
                        
                    fpath = os.path.join(dir, filename)

                    os.makedirs(dir, exist_ok=True)

                    def gen_bar_updater():
                        pbar = tqdm(total=None)

                        def bar_update(count, block_size, total_size):
                            if pbar.total is None and total_size:
                                pbar.total = total_size
                            progress_bytes = count * block_size
                            pbar.update(progress_bytes - pbar.n)

                        return bar_update

                    try:
                        print('Downloading ' + url + ' to ' + fpath)
                        urllib.request.urlretrieve(url, fpath, reporthook=gen_bar_updater())
                            
                    except (urllib.error.URLError, IOError) as e:
                        if url[:5] == 'https':
                            url = url.replace('https:', 'http:')
                            print('Failed download. Trying https -> http instead.'
                                    ' Downloading ' + url + ' to ' + fpath)
                            urllib.request.urlretrieve(url, fpath, reporthook=gen_bar_updater())
                        else:
                            raise e
                    # check integrity of downloaded file
                    if not check_integrity(fpath, None):
                        raise RuntimeError("File not found or corrupted.")

                print(f"Downloading `{self.tar_fpath}`")
                download_targz(
                    url=TARGZ_DOWNLOAD_URL, 
                    dir=str(self.tar_fpath.parent), 
                    filename=self.tar_fpath.name,
                )
            
            torch.save(self._prepare_original_data(), self.original_data_fpath)
            
            # save the md5 of the file generated above
            self._original_data_md5_fpath.write_text(
                hashlib.md5(self.original_data_fpath.read_bytes()).hexdigest()
            )
        
        print(f"Precomputing data for normal class {self.normal_class} for shape {self.shape} from {self.original_data_fpath}")
        original_dataset_dict = torch.load(self.original_data_fpath)
        
        # see the end of the function download_original_data() 
        size_interpolate = self.shape  # [height, width]

        dataset_dict = {
            'train_imgs': torch_interpolate(
                input=original_dataset_dict['train_imgs'], 
                size=size_interpolate,
            ).byte(), 
            'train_labels': original_dataset_dict['train_labels'],
            'train_anomaly_labels': original_dataset_dict['train_anomaly_labels'],
            'test_imgs': torch_interpolate(
                input=original_dataset_dict['test_imgs'], 
                size=size_interpolate,
            ).byte(), 
            'test_labels': original_dataset_dict['test_labels'],
            # gtmaps dont have the channel axis so it's added then removed again (unsqueeze and squeeze)
            'test_gtmaps': torch_interpolate(
                input=original_dataset_dict['test_gtmaps'], 
                size=size_interpolate,
            ).byte(), 
            'test_anomaly_labels': original_dataset_dict['test_anomaly_labels'],
            'anomaly_label_strings': original_dataset_dict['anomaly_label_strings'],
        }
        
        torch.save(dataset_dict, self.data_fpath)
        
        # save the md5 of the file generated above
        self._data_md5_fpath.write_text(
            hashlib.md5(self.data_fpath.read_bytes()).hexdigest()
        )
    
    @property
    def is_setup(self):
        if self._imgs is None:
            assert self._gtmaps is None, "_gtmaps is not None but _imgs is None"
            assert self._labels is None, "_labels is not None but _imgs is None"
            assert self._anomaly_labels is None, "_anomaly_labels is not None but _imgs is None"
            assert self._anomaly_label_strings is None, "_anomaly_label_strings is not None but _imgs is None"
            assert self._original_gtmaps is None, "_original_gtmaps is not None but _imgs is None"
            return False
        assert self._gtmaps is not None, "_gtmaps is None but _imgs is not None"
        assert self._labels is not None, "_labels is None but _imgs is not None"
        assert self._anomaly_labels is not None, "_anomaly_labels is None but _imgs is not None"
        assert self._anomaly_label_strings is not None, "_anomaly_label_strings is None but _imgs is not None"
        assert self._original_gtmaps is not None, "_original_gtmaps is None but _imgs is not None"        
        return True
            
    def setup(self):
        """load from the files"""
        
        if self.is_setup:
            print(f"Dataset already setup (split={self.split})")
            return

        print(f'Loading dataset from {self.data_fpath}...')
                
        dataset_dict = torch.load(self.data_fpath)
        anomaly_label_strings = dataset_dict['anomaly_label_strings']
        
        if self.split == SPLIT_TRAIN:
            imgs = dataset_dict['train_imgs']
            labels = dataset_dict['train_labels']
            imgs_shape = imgs.shape
            gtmaps_shape = (imgs_shape[0], 1,) + tuple(imgs_shape[2:])  # imgs have 3 channels, gtmaps just 1
            gtmaps = torch.zeros(gtmaps_shape).byte()
            anomaly_labels = dataset_dict['train_anomaly_labels']
            
            original_dataset_dict = torch.load(self.original_data_fpath)
            imgs_shape = original_dataset_dict["train_imgs"].shape
            gtmaps_shape = (imgs_shape[0], 1,) + tuple(imgs_shape[2:])  # imgs have 3 channels, gtmaps just 1
            original_gtmaps = torch.zeros(gtmaps_shape).byte()
            
        elif self.split == SPLIT_TEST:
            imgs = dataset_dict['test_imgs']
            labels = dataset_dict['test_labels']
            gtmaps = dataset_dict['test_gtmaps']
            anomaly_labels = dataset_dict['test_anomaly_labels']
            
            original_dataset_dict = torch.load(self.original_data_fpath)
            original_gtmaps = original_dataset_dict['test_gtmaps']
            
        else:
            raise ValueError(f'Unknown split {self.split}.')
        
        # =============================================================================
        # ================================ validation ================================
        # =============================================================================
        
        # ndim
        assert imgs.ndim == 4, f'Expected imgs to have 4 dimensions, got {imgs.ndim}'
        assert gtmaps.ndim == 4, f'Expected gtmaps to have 4 dimensions, got {gtmaps.ndim}'
        assert original_gtmaps.ndim == 4, f'Expected gtmaps to have 4 dimensions, got {original_gtmaps.ndim}'
        assert labels.ndim == 1, f'Expected labels to have 1 dimensions, got {labels.ndim}'
        assert anomaly_labels.ndim == 1, f'Expected anomaly_labels to have 1 dimensions, got {anomaly_labels.ndim}'
        
        # coherence of the shapes
        assert imgs.shape[2:] == self.shape, f'Expected imgs to have shape {self.shape}, got {imgs.shape[2:]}'
        assert imgs.shape[2:] == gtmaps.shape[2:], f'Expected imgs and gtmaps to have the same shape, got {imgs.shape} and {gtmaps.shape}'
        assert imgs.shape[0] == gtmaps.shape[0], f'Expected imgs and original_gtmaps to have the same number of samples, got {imgs.shape[0]} and {gtmaps.shape[0]}'
        assert imgs.shape[0] == original_gtmaps.shape[0], f'Expected imgs and original_gtmaps to have the same number of samples, got {imgs.shape[0]} and {original_gtmaps.shape[0]}'
        assert imgs.shape[0] == labels.shape[0], f'Expected imgs and labels to have the same number of samples, got {imgs.shape[0]} and {labels.shape[0]}'
        assert imgs.shape[0] == anomaly_labels.shape[0], f'Expected imgs and anomaly_labels to have the same number of samples, got {imgs.shape[0]} and {anomaly_labels.shape[0]}'
        
        # nchannels
        assert imgs.shape[1] == 3, f'Expected imgs to have 3 channels, got {imgs.shape[1]}'
        assert gtmaps.shape[1] == 1, f'Expected gtmaps to have 1 channel, got {gtmaps.shape[1]}'
        
        # dtype
        assert imgs.dtype == torch.uint8, f'Expected imgs to have dtype torch.uint8, got {imgs.dtype}'
        assert gtmaps.dtype == torch.uint8, f'Expected gtmaps to have dtype torch.uint8, got {gtmaps.dtype}'
        assert labels.dtype == torch.int32, f'Expected labels to have dtype torch.int32, got {labels.dtype}'
        assert anomaly_labels.dtype == torch.int32, f'Expected anomaly_labels to have dtype torch.int32, got {anomaly_labels.dtype}'
        
        # values
        assert tuple(sorted(gtmaps.unique().tolist())) in ((0, 255), (0,)), f"Expected gtmaps to have values (0, 255), got {tuple(sorted(gtmaps.unique().tolist()))}"
        assert tuple(sorted(labels.unique().tolist())) in tuple((c,) for c in range(NCLASSES)), f'Expected labels to have values 0 to {NCLASSES-1}, got {labels.unique()}'
        assert (labels == self.normal_class).all(), f'Expected labels to be all {self.normal_class}'
        
        if self.split == SPLIT_TRAIN:
            assert tuple(gtmaps.unique().tolist()) == (0,), f'Expected gtmaps to have values (0,), got {tuple(gtmaps.unique().tolist())}'
                    
        # =============================================================================
        imgs = imgs / 255.
        gtmaps = gtmaps / 255.
        original_gtmaps = original_gtmaps / 255.
        
        self.imgs = imgs
        self.gtmaps = gtmaps
        self.original_gtmaps = original_gtmaps
        self.labels = labels
        self.anomaly_labels = anomaly_labels
        self.anomaly_label_strings = anomaly_label_strings
                
        print('Dataset setup.')

    def __getitem__(self, index: int) -> Tuple[Tensor, int, Tensor]:
        
        if self.is_subset:
            index = self.subset_indices[index]
        
        # using clone() to avoid modifying the original tensors
        class_label = self.labels[index].clone()
        assert class_label == self.normal_class, f'Expected class label {self.normal_class} but got {class_label}'
        img= self.imgs[index].clone()
        gtmap = self.gtmaps[index].clone()
        anomaly_label = self.anomaly_labels[index].clone()
        
        # no more label_transform
        target = NOMINAL_TARGET if anomaly_label == NORMAL_LABEL_IDX else ANOMALY_TARGET

        return img, target, gtmap

    def __len__(self) -> int:
        return len(self.imgs) if not self.is_subset else len(self.subset_indices)

    def _prepare_original_data(self):
        
        with tempfile.TemporaryDirectory() as extract_dir:
            
            extract_dir = Path(extract_dir)
            
            print(f"Extracting {self.tar_fpath} to {extract_dir}")
            
            with tarfile.open(self.tar_fpath, 'r:xz') as tar:
                members_to_extract = [
                    tarinfo
                    for tarinfo in tar.getmembers()
                    if tarinfo.name.startswith(self.normal_class_label)
                ]
                tar.extractall(path=extract_dir, members=members_to_extract)

            print(f'Processing data for class {self.normal_class} ({self.normal_class_label})')
            
            train_imgs, train_labels, train_anomaly_labels = [], [], []
            test_imgs, test_labels, test_gtmaps, test_anomaly_labels = [], [], [], []
            
            # mapping of str -> int 
            # anomaly type name -> anomaly type id (0 up to n-1, where n-1 depends on the number of anomaly types of the normal class)
            anomaly_label_idmap = {NORMAL_LABEL: NORMAL_LABEL_IDX}  
            
            train_data_dpath: Path = extract_dir / self.normal_class_label / SPLIT_TRAIN
            
            test_data_dpath: Path = extract_dir / self.normal_class_label / SPLIT_TEST
            test_gtmap_dpath: Path = extract_dir / self.normal_class_label / "ground_truth"
            
            # only images
            train_fpaths = [
                img_fpath
                for anomaly_type_dpath in sorted(train_data_dpath.iterdir())  # there should be only "good"
                for img_fpath in sorted(anomaly_type_dpath.iterdir())
            ]
            
            def img_to_torch(img):
                return torch.from_numpy(np.array(img.convert('RGB'))).float().transpose(0, 2).transpose(1, 2).byte()
            
            for img_fpath in tqdm(train_fpaths, desc="train"):
                with img_fpath.open('rb') as f:
                    img = img_to_torch(Image.open(f))
                train_imgs.append(img)
                train_labels.append(self.normal_class)
                train_anomaly_labels.append(NORMAL_LABEL_IDX)
            
            train_imgs = torch.stack(train_imgs)
            train_labels = torch.IntTensor(train_labels)
            train_anomaly_labels = torch.IntTensor(train_anomaly_labels)
            
            # images and gtmaps
            test_fpath_dicts = [
                {
                    "img": img_fpath,
                    "gtmap": (
                        test_gtmap_dpath / anomaly_type_dpath.name / img_fpath.name.replace('.png', '_mask.png')
                        if anomaly_type_dpath.name != NORMAL_LABEL else
                        None
                    ),
                }
                for anomaly_type_dpath in sorted(test_data_dpath.iterdir())
                for img_fpath in sorted(anomaly_type_dpath.iterdir())
            ]
            
            for fpath_dict in tqdm(test_fpath_dicts, desc="test"):
                
                img_fpath = fpath_dict['img']
                gtmap_fpath = fpath_dict['gtmap']
                
                with img_fpath.open('rb') as f:
                    img = img_to_torch(Image.open(f))
            
                if gtmap_fpath is not None:
                    with gtmap_fpath.open('rb') as f:
                        mask = img_to_torch(Image.open(f))
                else:
                    mask = torch.zeros_like(img)    

                test_imgs.append(img)
                test_gtmaps.append(mask)
                
                anomaly_type = img_fpath.parent.name
                anomaly_type_idx = anomaly_label_idmap.setdefault(anomaly_type, len(anomaly_label_idmap))
                
                # these are the class labels
                test_labels.append(self.normal_class)
                # these are labels specific to the type of anomaly
                test_anomaly_labels.append(anomaly_type_idx)
                

            test_imgs = torch.stack(test_imgs)
            test_labels = torch.IntTensor(test_labels)
            test_gtmaps = torch.stack(test_gtmaps)[:, 0:1, :, :]  # r=g=b -> grayscale (0:1 makes it keep the dimension)
            test_anomaly_labels = torch.IntTensor(test_anomaly_labels)

            # invert the mapping to get the anomaly type name from the anomaly type id
            # int -> str
            anomaly_label_strings = list(zip(*sorted(anomaly_label_idmap.items(), key=lambda kv: kv[1])))[0]  # [0] will select the keys, [1] are the indices
            
        return {
            'train_imgs': train_imgs, 
            'train_labels': train_labels,
            'train_anomaly_labels': train_anomaly_labels,
            'test_imgs': test_imgs, 
            'test_labels': test_labels,
            'test_gtmaps': test_gtmaps, 
            'test_anomaly_labels': test_anomaly_labels,
            'anomaly_label_strings': anomaly_label_strings
        }


 # min max after gcn l1 norm has> been applied per class
_MIN_MAX_AFTER_GCN_L1NORM = [
    [
        (-1.3336724042892456, -1.3107913732528687, -1.2445921897888184),
        (1.3779616355895996, 1.3779616355895996, 1.3779616355895996),
    ],
    [
        (-2.2404820919036865, -2.3387579917907715, -2.2896201610565186),
        (4.573435306549072, 4.573435306549072, 4.573435306549072),
    ],
    [
        (-3.184587001800537, -3.164201259613037, -3.1392977237701416),
        (1.6995097398757935, 1.6011602878570557, 1.5209171772003174),
    ],
    [
        (-3.0334954261779785, -2.958242416381836, -2.7701096534729004),
        (6.503103256225586, 5.875098705291748, 5.814228057861328),
    ],
    [
        (-3.100773334503174, -3.100773334503174, -3.100773334503174),
        (4.27892541885376, 4.27892541885376, 4.27892541885376),
    ],
    [
        (-3.6565306186676025, -3.507692813873291, -2.7635035514831543),
        (18.966819763183594, 21.64590072631836, 26.408710479736328),
    ],
    [
        (-1.5192601680755615, -2.2068002223968506, -2.3948357105255127),
        (11.564697265625, 10.976534843444824, 10.378695487976074),
    ],
    [
        (-1.3207964897155762, -1.2889339923858643, -1.148416519165039),
        (6.854909896850586, 6.854909896850586, 6.854909896850586),
    ],
    [
        (-0.9883341193199158, -0.9822461605072021, -0.9288841485977173),
        (2.290637969970703, 2.4007883071899414, 2.3044068813323975),
    ],
    [
        (-7.236185073852539, -7.236185073852539, -7.236185073852539),
        (3.3777384757995605, 3.3777384757995605, 3.3777384757995605),
    ],
    [
        (-3.2036616802215576, -3.221003532409668, -3.305514335632324),
        (7.022546768188477, 6.115569114685059, 6.310940742492676),
    ],
    [
        (-0.8915618658065796, -0.8669204115867615, -0.8002046346664429),
        (4.4255571365356445, 4.642300128936768, 4.305730819702148),
    ],
    [
        (-1.9086798429489136, -2.0004451274871826, -1.929288387298584),
        (5.463134765625, 5.463134765625, 5.463134765625),
    ],
    [
        (-2.9547364711761475, -3.17536997795105, -3.143850803375244),
        (5.305514812469482, 4.535006523132324, 3.3618252277374268),
    ],
    [
        (-1.2906527519226074, -1.2906527519226074, -1.2906527519226074),
        (2.515115737915039, 2.515115737915039, 2.515115737915039),
    ],
]


def _lcn_global_normalization_transform(normal_class_: int):
    global _MIN_MAX_AFTER_GCN_L1NORM
    min_, max_ = _MIN_MAX_AFTER_GCN_L1NORM[normal_class_]
    return transforms.Normalize(min_, tuple(max__ - min__ for min__, max__ in zip(min_, max_)))


# mean and std of original images per class
_ORIGINAL_IMAGES_PERCLASS_MEAN = [
    (0.53453129529953, 0.5307118892669678, 0.5491130352020264),
    (0.326835036277771, 0.41494372487068176, 0.46718254685401917),
    (0.6953922510147095, 0.6663950085639954, 0.6533040404319763),
    (0.36377236247062683, 0.35087138414382935, 0.35671544075012207),
    (0.4484519958496094, 0.4484519958496094, 0.4484519958496094),
    (0.2390524297952652, 0.17620408535003662, 0.17206747829914093),
    (0.3919542133808136, 0.2631213963031769, 0.22006843984127045),
    (0.21368788182735443, 0.23478130996227264, 0.24079132080078125),
    (0.30240726470947266, 0.3029524087905884, 0.32861486077308655),
    (0.7099748849868774, 0.7099748849868774, 0.7099748849868774),
    (0.4567880630493164, 0.4711957275867462, 0.4482630491256714),
    (0.19987481832504272, 0.18578395247459412, 0.19361256062984467),
    (0.38699793815612793, 0.276934415102005, 0.24219433963298798),
    (0.6718143820762634, 0.47696375846862793, 0.35050269961357117),
    (0.4014520049095154, 0.4014520049095154, 0.4014520049095154)
]
_ORIGINAL_IMAGES_PERCLASS_STD = [
    (0.3667600452899933, 0.3666728734970093, 0.34991779923439026),
    (0.15321789681911469, 0.21510766446590424, 0.23905669152736664),
    (0.23858436942100525, 0.2591284513473511, 0.2601949870586395),
    (0.14506031572818756, 0.13994529843330383, 0.1276693195104599),
    (0.1636597216129303, 0.1636597216129303, 0.1636597216129303),
    (0.1688646823167801, 0.07597383111715317, 0.04383210837841034),
    (0.06069392338395119, 0.04061736911535263, 0.0303945429623127),
    (0.1602524220943451, 0.18222476541996002, 0.15336430072784424),
    (0.30409011244773865, 0.30411985516548157, 0.28656429052352905),
    (0.1337062269449234, 0.1337062269449234, 0.1337062269449234),
    (0.12076705694198608, 0.13341768085956573, 0.12879984080791473),
    (0.22920562326908112, 0.21501320600509644, 0.19536510109901428),
    (0.20621345937252045, 0.14321941137313843, 0.11695228517055511),
    (0.08259467780590057, 0.06751163303852081, 0.04756828024983406),
    (0.32304847240448, 0.32304847240448, 0.32304847240448)
]


def _normalize_mean_std(normal_class_: int):
    mean = _ORIGINAL_IMAGES_PERCLASS_MEAN[normal_class_]
    std = _ORIGINAL_IMAGES_PERCLASS_STD[normal_class_]
    return transforms.Normalize(mean, std)


def _clamp01():    
    return transforms.Lambda(lambda x: x.clamp(0, 1))
        

class EmbeddedPreprocessingDataset(VisionDataset, Dataset):
    """embed the preprocessing steps into the dataset --> allow to generate portable and consistent datasets when used out of lightning's framework"""
    
    def __init__(self, dataset, transform):
        self.dataset: LightningDataset = dataset
        self.transform = transform
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        batch = self.dataset[idx]
        batch = self.transform(batch)
        return batch


class MVTecAnomalyDetectionDataModule(LightningDataModule):
    
    @staticmethod       
    def _validate_batch_img(img_: Tensor):
        assert type(img_) == Tensor, f'Expected img to be a Tensor, got {type(img_)}'
        assert img_.ndim == 4, f'Expected img to have 4 dimensions, got {img_.ndim}'
        assert img_.shape[1] == 3, f'Expected img to have 3 channels, got {img_.shape[1]}'
        assert img_.dtype == torch.float32, f'Expected img to have dtype torch.float32, got {img_.dtype}'
        assert img_.min() >= 0, f'Expected img to have values >= 0, got {img_.min()}'
        assert img_.max() <= 1, f'Expected img to have values <= 1, got {img_.max()}'
        
    @staticmethod
    def _validate_batch_gtmap(gtmap_: Tensor):
        assert type(gtmap_) == Tensor, f'Expected gtmap_ to be a Tensor, got {type(gtmap_)}'
        assert gtmap_.ndim == 4, f'Expected gtmap_ to have 4 dimensions, got {gtmap_.ndim}'
        assert gtmap_.shape[1] == 1, f'Expected gtmap_ to have 1 channel, got {gtmap_.shape[1]}'
        assert gtmap_.dtype == torch.float32, f'Expected gtmap_ to have dtype torch.float32, got {gtmap_.dtype}'
        unique_gtmap_values = tuple(sorted(gtmap_.unique().tolist()))
        assert unique_gtmap_values in ((NOMINAL_TARGET, ANOMALY_TARGET), (NOMINAL_TARGET,)), f"Expected gtmap_ to have values in (0., 1.), got {unique_gtmap_values}"
        
    @staticmethod
    def _validate_batch_target(target_: Tensor):
        assert type(target_) == Tensor, f"target {target_} is not a Tensor"
        assert target_.ndim == 1, f'Expected target to have 1 dimension, got {target_.ndim}'
        assert target_.dtype == torch.int64, f'Expected target to have dtype torch.int64, got {target_.dtype}'
        unique_target_values = tuple(sorted(target_.unique().tolist()))
        assert unique_target_values in ((NOMINAL_TARGET, ANOMALY_TARGET), (NOMINAL_TARGET,), (ANOMALY_TARGET,)), f"Expected target to have values (0., 1.), got {unique_target_values}"
        
    @staticmethod       
    def _validate_batch_after_transform(transform: Callable):
        """this decorate the transforms to make sure they are not breaking basic stuff"""
    
        @functools.wraps(transform)
        def wrapper(*batches) -> Tuple[Tensor, Tensor, Tensor]:
            
            img_: Tensor
            target_: Tensor
            gtmap_: Tensor
            
            # the number of batches can be 1, 2, 3
            # 1: only img transform
            # 2: img and gtmap transform
            # 3: img, gtmap and target transform
            if len(batches) == 1:
                img_ = transform(*batches)
                ret = (img_,)
                
            else:
                ret = transform(*batches)
                assert len(ret) == len(batches), f"transform: {transform}: Expected transform to return {len(batches)} elements, got {len(ret)}"

                if len(ret) == 2:
                    img_, gtmap_ = ret
                    
                elif len(ret) == 3:
                    img_, target_, gtmap_ = ret
                
            MVTecAnomalyDetectionDataModule._validate_batch_img(img_)
            
            if len(ret) >= 2:
                MVTecAnomalyDetectionDataModule._validate_batch_gtmap(gtmap_)
                assert img_.shape[-2:] == gtmap_.shape[-2:], f'Expected img and gtmap_ to have the same shape, got {img_.shape} and {gtmap_.shape}'

            if len(ret) == 3:
                assert img_.shape[0] == target_.shape[0] == gtmap_.shape[0], f'Expected img, target and gtmap_ to have the same batch size, got {img_.shape[0]} and {target_.shape[0]} and {gtmap_.shape[0]}'
                MVTecAnomalyDetectionDataModule._validate_batch_target(target_)
                # target = 0 must not have 1s in the gtmap
                # target = 1 must have at least one
                assert torch.logical_or(
                    torch.logical_and(target_, gtmap_.sum(dim=(1, 2, 3)) > 0),
                    torch.logical_and(~target_, gtmap_.sum(dim=(1, 2, 3)) == 0),
                ).all()
                      
            if len(ret) == 1:
                return img_
            
            return ret
            
        return wrapper
    
    def __init__(
        self, 
        root: Path, 
        normal_class: int, 
        preprocessing: str, 
        preprocess_moment: str,
        supervise_mode: str, 
        batch_size: int,
        nworkers: int,
        pin_memory: bool,
        seed: int,
        raw_shape: Tuple[int, int], 
        net_shape: Tuple[int, int],
        real_anomaly_limit: int = None, 
    ):
        super().__init__()
                
        def validate_shape(shp):
            
            assert len(shp) == 2, "shape must be a tuple of length 3"
            
            width, height = shp
            assert width == height, f"width={width} != height={height}"

        validate_shape(raw_shape)
        validate_shape(net_shape)
                
        # validate preproc
        assert preprocessing in PREPROCESSING_CHOICES, f'`preproc` must be one of {PREPROCESSING_CHOICES}'
        assert supervise_mode in SUPERVISE_MODES, f'`supervise_mode` must be one of {SUPERVISE_MODES}'
        assert preprocess_moment in DATAMODULE_PREPROCESS_MOMENT_CHOICES, f'`preprocess_moment` must be one of {DATAMODULE_PREPROCESS_MOMENT_CHOICES}'

        if supervise_mode == SUPERVISE_MODE_REAL_ANOMALY:
            assert real_anomaly_limit is not None, f"'real_anomaly_limit' cannot be None if the supervise mode is {supervise_mode}"
            assert real_anomaly_limit > 0, f"real_anomaly_limit should be > 0, found {real_anomaly_limit}"

        assert nworkers >= 0, f"nworkers must be >= 0, found {nworkers}"
        
        # files
        self.root = root
        
        # modes and their subparameters        
        self.preprocessing = preprocessing
        self.supervise_mode = supervise_mode
        self.real_anomaly_limit = real_anomaly_limit
        self.preprocess_moment = preprocess_moment
        
        # data parameters
        self.raw_shape = tuple(raw_shape)  # it has to be a tuple for the comparison with torch shapes
        self.net_shape = tuple(net_shape)  
        self.normal_class = normal_class

        # training parameters
        self.batch_size = batch_size

        # processing parameters
        self.nworkers = nworkers
        self.pin_memory = pin_memory

        # randomness
        # seeds are validated in create_random_generator()
        self.seed = seed
        if self.supervise_mode == SUPERVISE_MODE_REAL_ANOMALY:
            self.real_anomaly_split_random_generator = create_numpy_random_generator(self.seed)
        
        # ========================================== TRANSFORMS ==========================================
        
        img_width_height = self.net_shape[-1]
        
        # use a function to instantiate it in order to make sure the function is the same but different instance
        def resize_netshape_multibatch():
            return make_multibatch(transforms.Resize(img_width_height, transforms.InterpolationMode.NEAREST))
        
        def randomcrop_multibatch():
            return make_multibatch_use_same_random_state(BatchRandomCrop(img_width_height, generator=create_numpy_random_generator(self.seed),))
        
        def randomly_resize_or_randomcrop_multibatch():
            return MultiBatchdRandomChoice(
                transforms=[randomcrop_multibatch(), resize_netshape_multibatch(),], 
                generator=create_numpy_random_generator(self.seed),
            )
        
        def randomflip_multibatch():
            return make_multibatch_use_same_random_state(
                BatchRandomFlip(vertical_probability=0.5, horizontal_probability=0.5, torch_generator=create_torch_random_generator(self.seed))
            )
        
        def randomaffine_multibatch():
            return make_multibatch_use_same_random_state(
                LoopBatchRandomAffine(
                    rotation_degrees=4,
                    scale_range=(0.95, 1.05),
                    shear_degrees=(4, 4),  # vertical and horizontal
                    torch_generator=create_torch_random_generator(self.seed),
                    interpolation=transforms.InterpolationMode.NEAREST,
                )
            )
        
        def _img_color_augmentation():
            return BatchCompose(transforms=[
                # my own implmentation of RandomChoice
                BatchRandomChoice(
                    transforms=[
                        transforms.ColorJitter(0.04, 0.04, 0.04, 0.04),
                        transforms.ColorJitter(0.005, 0.0005, 0.0005, 0.0005),
                    ], 
                    generator=create_numpy_random_generator(self.seed),
                ),
                BatchGaussianNoise(
                    mode=BatchGaussianNoise.STD_MODE_AUTO_IMAGEWISE, 
                    std_factor=.1,
                    generator=create_numpy_random_generator(self.seed),
                ),
            ])
        
        # ========================================== PREPROCESSING ==========================================
        # different types of preprocessing pipelines, 'lcn' is for using LCN, 'aug{X}' for augmentations
        if preprocessing == PREPROCESSING_LCNAUG1:
            
            # img and gtmap - train
            # my own implmentation of RandomChoice
            # multi compose is used because both gtmap and img must be resized
            self.train_img_and_gtmap_transform = randomly_resize_or_randomcrop_multibatch()
                        
            # img - train
            self.train_img_transform = BatchCompose(transforms=[
                _img_color_augmentation(),
                _clamp01(),
                BatchLocalContrastNormalization(), 
                _lcn_global_normalization_transform(normal_class),
                _clamp01(),
            ])
            
            # image and gtmap - test
            # multi compose is used because both gtmap and img must be resized
            self.test_img_and_gtmap_transform = resize_netshape_multibatch()
            
            # img - test
            self.test_img_transform = BatchCompose(transforms=[
                BatchLocalContrastNormalization(), 
                _lcn_global_normalization_transform(normal_class),
                _clamp01(),
            ])
        
        elif preprocessing == PREPROCESSING_LCNAUG2:
            
            # img and gtmap - train
            # my own implmentation of RandomChoice
            # multi compose is used because both gtmap and img must be resized by using the same transform
            self.train_img_and_gtmap_transform = MultiBatchCompose(transforms=[
                randomflip_multibatch(),
                randomaffine_multibatch(),
                randomly_resize_or_randomcrop_multibatch(),
            ])
                        
            # img - train
            self.train_img_transform = BatchCompose(transforms=[
                _img_color_augmentation(),
                _clamp01(),
                BatchLocalContrastNormalization(), 
                _lcn_global_normalization_transform(normal_class),
                _clamp01(),
            ])
            
            # image and gtmap - test
            # multi compose is used because both gtmap and img must be resized
            self.test_img_and_gtmap_transform = resize_netshape_multibatch()
            
            # img - test
            self.test_img_transform = BatchCompose(transforms=[
                BatchLocalContrastNormalization(), 
                _lcn_global_normalization_transform(normal_class),
                _clamp01(),
            ])
            
        else:
            raise ValueError(f'Preprocessing pipeline `{preprocessing}` is not known.')

        self.online_instance_replacer = BatchOnlineInstanceReplacer(
            supervise_mode=supervise_mode,
            generator=create_numpy_random_generator(self.seed),
            torch_generator=create_torch_random_generator(self.seed),
        )
        
        self.online_instance_replacer = self._validate_batch_after_transform(self.online_instance_replacer)
        self.train_img_and_gtmap_transform = self._validate_batch_after_transform(self.train_img_and_gtmap_transform)
        self.train_img_transform = self._validate_batch_after_transform(self.train_img_transform)
        self.test_img_and_gtmap_transform = self._validate_batch_after_transform(self.test_img_and_gtmap_transform)
        self.test_img_transform = self._validate_batch_after_transform(self.test_img_transform)
        
        self.train_mvtec = MvTec(
            root=str(self.root), 
            split=SPLIT_TRAIN, 
            shape=self.raw_shape, 
            normal_class=self.normal_class,
        )
        
        self.test_mvtec = MvTec(
            root=str(self.root), 
            split=SPLIT_TEST, 
            shape=self.raw_shape, 
            normal_class=self.normal_class,
        )

        self.save_hyperparameters()

    def prepare_data(self):
        self.train_mvtec.prepare_data()
        self.test_mvtec.prepare_data()

    def setup(self, stage: Optional[str] = None):
        
        # case 'validate' is necessary it uses the test split to validate the model
        if stage in (None, 'fit', 'validate'):
        
            if self.train_mvtec.is_setup:
                # avoid re-splitting the test set
                return             
            
            self.train_mvtec.setup()

            # 'validate' uses the test split to validate the model
            self.test_mvtec.setup()
            
            if self.supervise_mode == SUPERVISE_MODE_REAL_ANOMALY:
                
                # this is needed for the real anomaly split                
                print(f'using mode {self.supervise_mode} with real anomaly limit {self.real_anomaly_limit}, the anomalies used for training will be removed from the test set')
                
                n_anomaly_types = len(self.test_mvtec.anomaly_label_strings)
                anomaly_type_indices = list(range(n_anomaly_types))
                
                self.test_subsplits_indices_per_anomalytype = {}
                
                for idx in anomaly_type_indices:
                    
                    # indices of the images with the anomaly type `idx`
                    all_indices = np.argwhere(self.test_mvtec.anomaly_labels == idx).flatten()
                    
                    if idx == NORMAL_LABEL_IDX:  # 0
                        self.test_subsplits_indices_per_anomalytype[idx] = dict(
                            supervision=np.array([], dtype=int),
                            test=all_indices,
                        )
                        continue
                        
                    # those used in the online supervisor
                    supervision_indices = self.real_anomaly_split_random_generator.choice(
                        all_indices, size=min(self.real_anomaly_limit, all_indices.numel()), replace=False,
                    )
                    test_indices = np.array(sorted(set(all_indices) - set(supervision_indices)))

                    self.test_subsplits_indices_per_anomalytype[idx] = dict(
                        supervision=supervision_indices,
                        test=test_indices,
                    )
                
                self.test_subsplits_indices: Dict[str, List[int]] = dict(
                    supervision=np.sort(np.concatenate(tuple(
                        ss["supervision"] 
                        for ss in self.test_subsplits_indices_per_anomalytype.values()
                    ))).tolist(),
                    test=np.sort(np.concatenate(tuple(
                        ss["test"] for ss in self.test_subsplits_indices_per_anomalytype.values()
                    ))).tolist(),
                )
                
                # create a copy with all none cuz the train transforms are already applied
                self.real_anomaly_mvtec = MvTec(
                    root=self.test_mvtec.root,
                    split=self.test_mvtec.split,
                    normal_class=self.test_mvtec.normal_class,
                    shape=self.test_mvtec.shape,
                )
                # no need to prepare data, it's already done in the test set
                self.real_anomaly_mvtec.setup()
                self.real_anomaly_mvtec.subset_indices = self.test_subsplits_indices["supervision"]
                self.test_mvtec.subset_indices = self.test_subsplits_indices["test"]
                
                # __wrapped__ is needed for the online instance replacer because of _validate_batch_after_transform
                self.online_instance_replacer.__wrapped__.real_anomaly_dataloader = DataLoader(
                    dataset=self.real_anomaly_mvtec, 
                    batch_size=self.batch_size,
                    shuffle=True,
                    num_workers=0, 
                    pin_memory=False, 
                    persistent_workers=False,
                )

        elif stage == 'test':
            
            if self.test_mvtec.is_setup:
                return

            self.test_mvtec.setup()
        
        else:
            raise ValueError(f'Unknown stage {stage}')
    
    def on_before_batch_transfer(self, batch, dataloader_idx: int):
        
        img, target, gtmap = batch
                
        if self.trainer.training:
            img, target, gtmap = self.online_instance_replacer(img, target, gtmap)
        
        if self.preprocess_moment != DATAMODULE_PREPROCESS_MOMENT_BEFORE_BATCH_TRANSFER:
            return img, target, gtmap
        
        if self.trainer.training:
            img, gtmap = self.train_img_and_gtmap_transform(img, gtmap)
            img = self.train_img_transform(img)
            return img, target, gtmap
        
        # elif self.trainer.testing or self.trainer.validating or self.trainer.sanity_checking:
        elif self.trainer.testing or self.trainer.validating:
            img, gtmap = self.test_img_and_gtmap_transform(img, gtmap)
            img = self.test_img_transform(img)
            return img, target, gtmap
        
        else:
            raise NotImplementedError(f'Unknown stage. {self.trainer.state}')
        
    def on_after_batch_transfer(self, batch, dataloader_idx: int):

        img, target, gtmap = batch
        
        if self.preprocess_moment != DATAMODULE_PREPROCESS_MOMENT_AFTER_BATCH_TRANSFER:
            return img, target, gtmap
        
        if self.trainer.training:
            img, gtmap = self.train_img_and_gtmap_transform(img, gtmap)
            img = self.train_img_transform(img)
            return img, target, gtmap
        
        # elif self.trainer.testing or self.trainer.validating or self.trainer.sanity_checking:
        elif self.trainer.testing or self.trainer.validating:
            img, gtmap = self.test_img_and_gtmap_transform(img, gtmap)
            img = self.test_img_transform(img)
            return img, target, gtmap
        
        else:
            raise NotImplementedError(f'Unknown stage. {self.trainer.state}')
    
    def train_dataloader(self, batch_size_override: Optional[int] = None, nworkers_override: Optional[int] = None, embed_preprocessing: bool = False):
        """batch_size_override: override the batch size for special purposes like generating the preview faster"""
        
        batchsize = batch_size_override if batch_size_override is not None else self.batch_size
        nworkers = nworkers_override if nworkers_override is not None else self.nworkers
        dataset = self.train_mvtec
        
        if embed_preprocessing:
            
            def preprocessing(batch):
                """this goes inside the dataloader, so the batch should actually be a single image"""
                
                img: Tensor 
                target: int
                gtmap: Tensor
                img, target, gtmap = batch

                # create a fake batch dimension for the transforms
                img = img.unsqueeze(0)
                target = torch.tensor([target], dtype=torch.int64)
                gtmap = gtmap.unsqueeze(0)

                img, target, gtmap = self.online_instance_replacer(img, target, gtmap)
                img, gtmap = self.train_img_and_gtmap_transform(img, gtmap)
                img = self.train_img_transform(img)
                
                return img[0], int(target[0]), gtmap[0]
            
            dataset = EmbeddedPreprocessingDataset(dataset=dataset, transform=preprocessing,)
        
        generator = torch.Generator()    
        generator.manual_seed(self.seed)
        
        return DataLoader(
            dataset=dataset, 
            batch_size=batchsize, 
            shuffle=True,
            num_workers=nworkers, 
            pin_memory=self.pin_memory, 
            persistent_workers=False,
            generator=generator,
        )
        
    def val_dataloader(self, *args, **kwargs):
        return self.test_dataloader(*args, **kwargs)

    def test_dataloader(self, batch_size_override: Optional[int] = None, nworkers_override: Optional[int] = None, embed_preprocessing: bool = False):
        """batch_size_override: override the batch size for special purposes like generating the preview faster"""
        
        batchsize = batch_size_override if batch_size_override is not None else self.batch_size
        nworkers = nworkers_override if nworkers_override is not None else self.nworkers
        dataset = self.test_mvtec
        
        if embed_preprocessing:
            
            def preprocessing(batch):
                
                img: Tensor 
                target: int
                gtmap: Tensor
                img, target, gtmap = batch

                # create a fake batch dimension for the transforms
                img = img.unsqueeze(0)
                target = torch.tensor([target], dtype=torch.int64)
                gtmap = gtmap.unsqueeze(0)
                
                img, gtmap = self.test_img_and_gtmap_transform(img, gtmap)
                img = self.test_img_transform(img)
                
                return img[0], int(target[0]), gtmap[0]
            
            dataset = EmbeddedPreprocessingDataset(
                dataset=dataset,
                transform=preprocessing,
            )
            
        generator = torch.Generator()    
        generator.manual_seed(self.seed)
        
        return DataLoader(
            dataset=dataset, 
            batch_size=batchsize, 
            shuffle=False,
            num_workers=nworkers, 
            pin_memory=self.pin_memory, 
            persistent_workers=False,
            generator=generator,
        )
        

if __name__ == "__main__":
    
    datamodule = MVTecAnomalyDetectionDataModule(
        root="../../data/datasets",
        normal_class=0,
        preprocessing=PREPROCESSING_LCNAUG1,
        supervise_mode=SUPERVISE_MODE_SYNTHETIC_ANOMALY_CONFETTI,
        # supervise_mode=SUPERVISE_MODE_REAL_ANOMALY,
        batch_size=128,
        nworkers=0,
        pin_memory=False,
        raw_shape=(260, 260),
        net_shape=(224, 224),
        real_anomaly_limit=1,
        seed=0,
    )
    
    datamodule.prepare_data()

    savedir = Path.home() / "fcdd/data/tmp"
    savedir.mkdir(exist_ok=True)
    
    preview_nimages = 20
    
    # train
    datamodule.setup("fit")

    norm_imgs, norm_gtmaps, anom_imgs, anom_gtmaps = generate_dataloader_images(
        datamodule.train_dataloader(batch_size_override=2 * preview_nimages, embed_preprocessing=True), 
        nimages_perclass=preview_nimages
    )
    train_preview_fig = generate_dataloader_preview_single_fig(norm_imgs, norm_gtmaps, anom_imgs, anom_gtmaps, )      
    train_preview_fig.savefig(
        fname=savedir / f"train.{train_preview_fig.label}.png",
        # this makes sure that the number of pixels in the image is exactly the 
        # same as the number of pixels in the tensors 
        # (other conditions in the functions that generate the images)
        dpi=1, 
        pad_inches=0, bbox_inches='tight', 
    )
    
    # test
    datamodule.setup("test")

    norm_imgs, norm_gtmaps, anom_imgs, anom_gtmaps = generate_dataloader_images(
        datamodule.test_dataloader(batch_size_override=2 * preview_nimages, embed_preprocessing=True), 
        nimages_perclass=preview_nimages
    )
    test_preview_fig = generate_dataloader_preview_single_fig(norm_imgs, norm_gtmaps, anom_imgs, anom_gtmaps, )      
    test_preview_fig.savefig(
        fname=savedir / f"test.{test_preview_fig.label}.png",
        # this makes sure that the number of pixels in the image is exactly the 
        # same as the number of pixels in the tensors 
        # (other conditions in the functions that generate the images)
        dpi=1, 
        pad_inches=0, bbox_inches='tight', 
    )
    