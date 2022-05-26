import hashlib
import os
import random
import tarfile
import tempfile
import traceback
from copy import deepcopy
from itertools import cycle
from pathlib import Path
from typing import Callable, List, Tuple

import numpy as np
import torch
import torchvision.transforms as transforms
from kornia import gaussian_blur2d
from PIL import Image
from pytorch_lightning import LightningDataModule
from six.moves import urllib
from skimage.transform import rotate as im_rotate
from torch.nn.functional import interpolate as torch_interpolate
from torch.utils.data import DataLoader, Subset
from torch.utils.data.dataset import Dataset
from torchvision.datasets import VisionDataset
from torchvision.datasets.imagenet import check_integrity
from tqdm import tqdm

from data_dev01 import (ImgGtmapLabelTransform, ImgGtmapTensorsToUint8,
                        ImgGtmapToPIL, MultiCompose, NOMINAL_TARGET, ANOMALY_TARGET, generate_dataloader_images, generate_dataloader_preview_multiple_fig, generate_dataloader_preview_single_fig)

CLASSES_LABELS = (
    'bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather',
    'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor',
    'wood', 'zipper'
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
PREPROCESSING_CHOICES = (PREPROCESSING_LCNAUG1,)

TARGZ_DOWNLOAD_URL = "ftp://guest:GU%2E205dldo@ftp.softronics.ch/mvtec_anomaly_detection/mvtec_anomaly_detection.tar.xz"
TARGZ_FNAME = 'mvtec_anomaly_detection.tar.xz'

BASE_FOLDER = 'mvtec'


def confetti_noise(
    size: torch.Size, 
    p: float = 0.01,
    blobshaperange: Tuple[Tuple[int, int], Tuple[int, int]] = ((3, 3), (5, 5)),
    fillval: int = 255, 
    backval: int = 0, 
    ensureblob: bool = True, 
    awgn: float = 0.0,
    clamp: bool = False, 
    onlysquared: bool = True, 
    rotation: int = 0,
    colorrange: Tuple[int, int] = None
) -> torch.Tensor:
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
    assert len(size) == 4 or len(size) == 3, 'size must be n x c x h x w'
    if isinstance(blobshaperange[0], int) and isinstance(blobshaperange[1], int):
        blobshaperange = (blobshaperange, blobshaperange)
    assert len(blobshaperange) == 2
    assert len(blobshaperange[0]) == 2 and len(blobshaperange[1]) == 2
    assert colorrange is None or len(size) == 4 and size[1] == 3
    out_size = size
    colors = []
    if len(size) == 3:
        size = (size[0], 1, size[1], size[2])  # add channel dimension
    else:
        size = tuple(size)  # Tensor(torch.size) -> tensor of shape size, Tensor((x, y)) -> Tensor with 2 elements x & y
    mask = (torch.rand((size[0], size[2], size[3])) < p).unsqueeze(1)  # mask[i, j, k] == 1 for center of blob
    while ensureblob and (mask.view(mask.size(0), -1).sum(1).min() == 0):
        idx = (mask.view(mask.size(0), -1).sum(1) == 0).nonzero().squeeze()
        s = idx.size(0) if len(idx.shape) > 0 else 1
        mask[idx] = (torch.rand((s, 1, size[2], size[3])) < p)
    res = torch.empty(size).fill_(backval).int()
    idx = mask.nonzero()  # [(idn, idz, idy, idx), ...] = indices of blob centers
    if idx.reshape(-1).size(0) == 0:
        return torch.zeros(out_size).int()

    all_shps = [
        (x, y) for x in range(blobshaperange[0][0], blobshaperange[1][0] + 1)
        for y in range(blobshaperange[0][1], blobshaperange[1][1] + 1) if not onlysquared or x == y
    ]
    picks = torch.FloatTensor(idx.size(0)).uniform_(0, len(all_shps)).int()  # for each blob center pick a shape
    nidx = []
    for n, blobshape in enumerate(all_shps):
        if (picks == n).sum() < 1:
            continue
        bhs = range(-(blobshape[0] // 2) if blobshape[0] % 2 != 0 else -(blobshape[0] // 2) + 1, blobshape[0] // 2 + 1)
        bws = range(-(blobshape[1] // 2) if blobshape[1] % 2 != 0 else -(blobshape[1] // 2) + 1, blobshape[1] // 2 + 1)
        extends = torch.stack([
            torch.zeros(len(bhs) * len(bws)).long(),
            torch.zeros(len(bhs) * len(bws)).long(),
            torch.arange(bhs.start, bhs.stop).repeat(len(bws)),
            torch.arange(bws.start, bws.stop).unsqueeze(1).repeat(1, len(bhs)).reshape(-1)
        ]).transpose(0, 1)
        nid = idx[picks == n].unsqueeze(1) + extends.unsqueeze(0)
        if colorrange is not None:
            col = torch.randint(
                colorrange[0], colorrange[1], (3, )
            )[:, None].repeat(1, nid.reshape(-1, nid.size(-1)).size(0)).int()
            colors.append(col)
        nid = nid.reshape(-1, extends.size(1))
        nid = torch.max(torch.min(nid, torch.LongTensor(size) - 1), torch.LongTensor([0, 0, 0, 0]))
        nidx.append(nid)
    idx = torch.cat(nidx)  # all pixel indices that blobs cover, not only center indices
    shp = res[idx.transpose(0, 1).numpy()].shape
    if colorrange is not None:
        colors = torch.cat(colors, dim=1)
        gnoise = (torch.randn(3, *shp) * awgn).int() if awgn != 0 else (0, 0, 0)
        res[idx.transpose(0, 1).numpy()] = colors[0] + gnoise[0]
        res[(idx + torch.LongTensor((0, 1, 0, 0))).transpose(0, 1).numpy()] = colors[1] + gnoise[1]
        res[(idx + torch.LongTensor((0, 2, 0, 0))).transpose(0, 1).numpy()] = colors[2] + gnoise[2]
    else:
        gnoise = (torch.randn(shp) * awgn).int() if awgn != 0 else 0
        res[idx.transpose(0, 1).numpy()] = torch.ones(shp).int() * fillval + gnoise
        res = res[:, 0, :, :]
        if len(out_size) == 4:
            res = res.unsqueeze(1).repeat(1, out_size[1], 1, 1)
    if clamp:
        res = res.clamp(backval, fillval) if backval < fillval else res.clamp(fillval, backval)
    mask = mask[:, 0, :, :]
    if rotation > 0:
        
        def ceil(x: float):
            return int(np.ceil(x))
        
        def floor(x: float):
            return int(np.floor(x))
        
        idx = mask.nonzero()
        res = res.unsqueeze(1) if res.dim() != 4 else res
        res = res.transpose(1, 3).transpose(1, 2)
        for pick, blbctr in zip(picks, mask.nonzero()):
            rot = np.random.uniform(-rotation, rotation)
            p1, p2 = all_shps[pick]
            dims = (
                blbctr[0],
                slice(max(blbctr[1] - floor(0.75 * p1), 0), min(blbctr[1] + ceil(0.75 * p1), res.size(1) - 1)),
                slice(max(blbctr[2] - floor(0.75 * p2), 0), min(blbctr[2] + ceil(0.75 * p2), res.size(2) - 1)),
                ...
            )
            res[dims] = torch.from_numpy(
                im_rotate(
                    res[dims].float(), rot, order=0, cval=0, center=(blbctr[1]-dims[1].start, blbctr[2]-dims[2].start),
                    clip=False
                )
            ).int()
        res = res.transpose(1, 2).transpose(1, 3)
        res = res.squeeze() if len(out_size) != 4 else res
    return res


def local_contrast_normalization(x: torch.tensor):
    """
    Apply local contrast normalization to tensor, i.e. subtract mean across features (pixels) and normalize by the the standard deviation with L1- across features (pixels).
    Note this is a *per sample* normalization globally across features (and not across the dataset).
    """
    n_features = int(np.prod(x.shape))
    x = x - torch.mean(x)  # mean over all features (pixels) per sample
    x_scale = torch.mean(torch.abs(x))
    x /= (x_scale if x_scale != 0 else 1)
    return x


def merge_image_and_synthetic_noise(
        img: torch.Tensor, 
        gt: torch.Tensor, 
        generated_noise: torch.Tensor, 
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
    

class OnlineInstanceReplacer(ImgGtmapLabelTransform):
    
    invert_threshold = 0.025

    def __init__(
        self, 
        supervise_mode: str, 
        p: float = 0.5, 
    ):
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
        self.supervise_mode = supervise_mode
        self._real_anomaly_dataloader = None
        self.p = p
        
        if self.supervise_mode == SUPERVISE_MODE_REAL_ANOMALY:
            pass
            
        elif self.supervise_mode == SUPERVISE_MODE_SYNTHETIC_ANOMALY_CONFETTI:
            pass
        
        else:
            raise NotImplementedError(f'Supervise mode `{self.supervise_mode}` unknown.')        
    
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
    
    def __call__(self, img: torch.Tensor, gtmap: torch.Tensor, label: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Based on the probability defined in __init__, replaces (img, gt, target) with an artificial anomaly.
        :param img: some torch tensor image
        :param gt: some ground-truth map (can be None)
        :param target: some label
        :return: (img, gt, target)
        """
        if random.random() < self.p:
            return self.replace(img, gtmap, label)
            
        return img, gtmap, label

    def replace(self, img: torch.Tensor, gtmap: torch.Tensor, label: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        
        img = img.unsqueeze(0) if img is not None else img
        
        # gt value 1 will be put to anom_label in mvtec_bases get_item
        gtmap = gtmap.unsqueeze(0).fill_(1).float() if gtmap is not None else gtmap
        
        if self.supervise_mode == SUPERVISE_MODE_SYNTHETIC_ANOMALY_CONFETTI:
             
            generated_noise_rgb = confetti_noise(
                img.shape, 
                0.000018, 
                ((8, 8), (54, 54)), 
                fillval=255, 
                clamp=False, 
                awgn=0, 
                rotation=45, 
                colorrange=(-256, 0),
            )
            generated_noise = confetti_noise(
                img.shape, 
                0.000012, 
                ((8, 8), (54, 54)), 
                fillval=-255, 
                clamp=False, 
                awgn=0, 
                rotation=45,
            )
            generated_noise = generated_noise_rgb + generated_noise
            # generated_noise = smooth_noise(generated_noise, 25, 5, 1.0)
            
            # img, gtmap = merge_image_and_synthetic_noise(
            img, gtmap = merge_image_and_synthetic_noise(
                (img * 255).int(), 
                gtmap.squeeze(), 
                generated_noise, 
                invert_threshold=self.invert_threshold
            )
            img = img.float() / 255.0
            return img[0], gtmap[0], ANOMALY_TARGET
            
        elif self.supervise_mode == SUPERVISE_MODE_REAL_ANOMALY:
            
            img_, target, gtmap_ = next(self.real_anomaly_dataloader)

            assert img_.shape == img.shape, f"img.shape {img.shape} != img_.shape {img_.shape}"
            assert gtmap_.shape == gtmap.shape, f"gtmap.shape {gtmap.shape} != gtmap_.shape {gtmap_.shape}"
            
            img = img_
            gtmap = gtmap_

            assert target.shape == (1,), f"target.shape should be (1,), but is {target.shape}"
            assert target[0] == ANOMALY_TARGET, f"target should be {ANOMALY_TARGET}"
            
            # img = anom_img.clamp(0, 255).byte() 
            return img[0], gtmap[0], ANOMALY_TARGET  # [0] gets rid of the batch idx axis
        
        else:
            raise NotImplementedError('Supervise mode {self.supervise_mode} unknown.')
        
    
class MvTec(VisionDataset, Dataset):
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
        :param enlarge: whether to enlarge the dataset, i.e. repeat all data samples ten times.
            Consequently, one iteration (epoch) of the data loader returns ten times as many samples.
            This speeds up loading because the MVTec-AD dataset has a poor number of samples and
            PyTorch requires additional work in between epochs.
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
    
    @property
    def normal_class_label(self):
        return CLASSES_LABELS[self.normal_class]
    
    @property
    def imgs(self):
        if self._imgs is None:
            raise RuntimeError("MVTec dataset not setup yet. Please call setup() first.")
        return self._imgs        
    
    @property
    def labels(self):
        if self._labels is None:
            raise RuntimeError("MVTec dataset not setup yet. Please call setup() first.")
        return self._labels        
    
    @property
    def gtmaps(self):
        if self._gtmaps is None:
            raise RuntimeError("MVTec dataset not setup yet. Please call setup() first.")
        return self._gtmaps    
    
    @property
    def anomaly_labels(self):
        if self._anomaly_labels is None:
            raise RuntimeError("MVTec dataset not setup yet. Please call setup() first.")
        return self._anomaly_labels        
    
    @property
    def anomaly_label_strings(self):
        if self._anomaly_label_strings is None:
            raise RuntimeError("MVTec dataset not setup yet. Please call setup() first.")
        return self._anomaly_label_strings        
    
    @property
    def original_gtmaps(self):
        if self._original_gtmaps is None:
            raise RuntimeError("MVTec dataset not setup yet. Please call setup() first.")
        return self._original_gtmaps               
    
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
            
    def setup(self):
        """load from the files"""

        print(f'Loading dataset from {self.data_fpath}...')
                
        dataset_dict = torch.load(self.data_fpath)
        self._anomaly_label_strings = dataset_dict['anomaly_label_strings']
        
        if self.split == SPLIT_TRAIN:
            self._imgs = dataset_dict['train_imgs']
            self._labels = dataset_dict['train_labels']
            imgs_shape = self.imgs.shape
            gtmaps_shape = (imgs_shape[0], 1,) + tuple(imgs_shape[2:])  # imgs have 3 channels, gtmaps just 1
            self._gtmaps = torch.zeros(gtmaps_shape).byte()
            self._anomaly_labels = dataset_dict['train_anomaly_labels']
            
            original_dataset_dict = torch.load(self.original_data_fpath)
            imgs_shape = original_dataset_dict["train_imgs"].shape
            gtmaps_shape = (imgs_shape[0], 1,) + tuple(imgs_shape[2:])  # imgs have 3 channels, gtmaps just 1
            self._original_gtmaps = torch.zeros(gtmaps_shape).byte()
            
        elif self.split == SPLIT_TEST:
            self._imgs = dataset_dict['test_imgs']
            self._labels = dataset_dict['test_labels']
            self._gtmaps = dataset_dict['test_gtmaps']
            self._anomaly_labels = dataset_dict['test_anomaly_labels']
            
            original_dataset_dict = torch.load(self.original_data_fpath)
            self._original_gtmaps = original_dataset_dict['test_gtmaps']
            
        else:
            raise ValueError(f'Unknown split {self.split}.')
        
        # =============================================================================
        # ================================ validation ================================
        # =============================================================================
        
        # ndim
        assert self.imgs.ndim == 4, f'Expected imgs to have 4 dimensions, got {self.imgs.ndim}'
        assert self.gtmaps.ndim == 4, f'Expected gtmaps to have 4 dimensions, got {self.gtmaps.ndim}'
        assert self.original_gtmaps.ndim == 4, f'Expected gtmaps to have 4 dimensions, got {self.original_gtmaps.ndim}'
        assert self.labels.ndim == 1, f'Expected labels to have 1 dimensions, got {self.labels.ndim}'
        assert self.anomaly_labels.ndim == 1, f'Expected anomaly_labels to have 1 dimensions, got {self.anomaly_labels.ndim}'
        
        # coherence of the shapes
        assert self.imgs.shape[2:] == self.shape, f'Expected imgs to have shape {self.shape}, got {self.imgs.shape[2:]}'
        assert self.imgs.shape[2:] == self.gtmaps.shape[2:], f'Expected imgs and gtmaps to have the same shape, got {self.imgs.shape} and {self.gtmaps.shape}'
        assert self.imgs.shape[0] == self.gtmaps.shape[0], f'Expected imgs and original_gtmaps to have the same number of samples, got {self.imgs.shape[0]} and {self.gtmaps.shape[0]}'
        assert self.imgs.shape[0] == self.original_gtmaps.shape[0], f'Expected imgs and original_gtmaps to have the same number of samples, got {self.imgs.shape[0]} and {self.original_gtmaps.shape[0]}'
        assert self.imgs.shape[0] == self.labels.shape[0], f'Expected imgs and labels to have the same number of samples, got {self.imgs.shape[0]} and {self.labels.shape[0]}'
        assert self.imgs.shape[0] == self.anomaly_labels.shape[0], f'Expected imgs and anomaly_labels to have the same number of samples, got {self.imgs.shape[0]} and {self.anomaly_labels.shape[0]}'
        
        # nchannels
        assert self.imgs.shape[1] == 3, f'Expected imgs to have 3 channels, got {self.imgs.shape[1]}'
        assert self.gtmaps.shape[1] == 1, f'Expected gtmaps to have 1 channel, got {self.gtmaps.shape[1]}'
        
        # dtype
        assert self.imgs.dtype == torch.uint8, f'Expected imgs to have dtype torch.uint8, got {self.imgs.dtype}'
        assert self.gtmaps.dtype == torch.uint8, f'Expected gtmaps to have dtype torch.uint8, got {self.gtmaps.dtype}'
        assert self.labels.dtype == torch.int32, f'Expected labels to have dtype torch.int32, got {self.labels.dtype}'
        assert self.anomaly_labels.dtype == torch.int32, f'Expected anomaly_labels to have dtype torch.int32, got {self.anomaly_labels.dtype}'
        
        # values
        assert tuple(sorted(self.gtmaps.unique().tolist())) in ((0, 255), (0,)), f"Expected gtmaps to have values (0, 255), got {tuple(sorted(self.gtmaps.unique().tolist()))}"
        assert tuple(sorted(self.labels.unique().tolist())) in tuple((c,) for c in range(NCLASSES)), f'Expected labels to have values 0 to {NCLASSES-1}, got {self.labels.unique()}'
        assert (self.labels == self.normal_class).all(), f'Expected labels to be all {self.normal_class}'
        
        # =============================================================================
        self._imgs = self._imgs / 255.
        self._gtmaps = self._gtmaps / 255.
        self._original_gtmaps = self._original_gtmaps / 255.
        
        print('Dataset setup.')

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int, torch.Tensor]:
        
        img, class_label, anomaly_label = self.imgs[index], self.labels[index], self.anomaly_labels[index]
        
        assert class_label == self.normal_class, f'Expected class label {self.normal_class} but got {class_label}'
        gtmap = self.gtmaps[index]
        
        # no more label_transform
        target = NOMINAL_TARGET if anomaly_label == NORMAL_LABEL_IDX else ANOMALY_TARGET

        def validate(img_: torch.Tensor, gtmap_: torch.Tensor, target_: int) -> None:
            """make sure things are as expected after each transformation"""
            
            # type
            assert type(img_) == torch.Tensor, f'Expected img to be a torch.Tensor, got {type(img_)}'
            assert type(gtmap_) == torch.Tensor, f'Expected gtmap to be a torch.Tensor, got {type(gtmap_)}'
            assert type(target_) == int, f'Expected target to be an int, got {type(target_)}'
                        
            # ndim
            assert img_.ndim == 3, f'Expected img to have 3 dimensions, got {img_.ndim}'
            assert gtmap_.ndim == 3, f'Expected gtmap to have 3 dimensions, got {gtmap_.ndim}'
            
            # coherence of the shapes
            assert img_.shape[1:] == gtmap_.shape[1:], f'Expected img and gtmap to have the same shape, got {img_.shape} and {gtmap_.shape}'
            
            # nchannels
            assert img_.shape[0] == 3, f'Expected img to have 3 channels, got {img_.shape[0]}'
            assert gtmap_.shape[0] == 1, f'Expected gtmap to have 1 channel, got {gtmap_.shape[0]}'
            
            # dtype
            assert img_.dtype == torch.float32, f'Expected img to have dtype torch.float32, got {img_.dtype}'
            assert gtmap_.dtype == torch.float32, f'Expected gtmap to have dtype torch.float32, got {gtmap_.dtype}'
            
            # values
            assert target_ in (NOMINAL_TARGET, ANOMALY_TARGET), f'Expected target to be either {NOMINAL_TARGET} or {ANOMALY_TARGET}, got {target_}'
            assert img_.min() >= 0, f'Expected img to have values >= 0, got {img_.min()}'
            assert img_.max() <= 1, f'Expected img to have values <= 1, got {img_.max()}'
            assert tuple(sorted(gtmap_.unique().tolist())) in ((0., 1.), (0.,)), f"Expected gtmap to have values (0., 1.), got {tuple(sorted(gtmap.unique().tolist()))}"

        # this try/catch makes it easier to debug because you cans see self
        validate(img, gtmap, target)

        if self.all_transform is not None:
            img, gtmap, target = self.all_transform((img, gtmap, target))
            validate(img, gtmap, target)

        if self.img_and_gtmap_transform is not None:
            img, gtmap = self.img_and_gtmap_transform((img, gtmap))
            validate(img, gtmap, target)
            
        if self.img_transform is not None:
            img = self.img_transform(img)
            validate(img, gtmap, target)
        
        return img, target, gtmap

    def __len__(self) -> int:
        return len(self.imgs)

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

 # min max after gcn l1 norm has> been applied
min_max_l1 = [
    [(-1.3336724042892456, -1.3107913732528687, -1.2445921897888184),
        (1.3779616355895996, 1.3779616355895996, 1.3779616355895996)],
    [(-2.2404820919036865, -2.3387579917907715, -2.2896201610565186),
        (4.573435306549072, 4.573435306549072, 4.573435306549072)],
    [(-3.184587001800537, -3.164201259613037, -3.1392977237701416),
        (1.6995097398757935, 1.6011602878570557, 1.5209171772003174)],
    [(-3.0334954261779785, -2.958242416381836, -2.7701096534729004),
        (6.503103256225586, 5.875098705291748, 5.814228057861328)],
    [(-3.100773334503174, -3.100773334503174, -3.100773334503174),
        (4.27892541885376, 4.27892541885376, 4.27892541885376)],
    [(-3.6565306186676025, -3.507692813873291, -2.7635035514831543),
        (18.966819763183594, 21.64590072631836, 26.408710479736328)],
    [(-1.5192601680755615, -2.2068002223968506, -2.3948357105255127),
        (11.564697265625, 10.976534843444824, 10.378695487976074)],
    [(-1.3207964897155762, -1.2889339923858643, -1.148416519165039),
        (6.854909896850586, 6.854909896850586, 6.854909896850586)],
    [(-0.9883341193199158, -0.9822461605072021, -0.9288841485977173),
        (2.290637969970703, 2.4007883071899414, 2.3044068813323975)],
    [(-7.236185073852539, -7.236185073852539, -7.236185073852539),
        (3.3777384757995605, 3.3777384757995605, 3.3777384757995605)],
    [(-3.2036616802215576, -3.221003532409668, -3.305514335632324),
        (7.022546768188477, 6.115569114685059, 6.310940742492676)],
    [(-0.8915618658065796, -0.8669204115867615, -0.8002046346664429),
        (4.4255571365356445, 4.642300128936768, 4.305730819702148)],
    [(-1.9086798429489136, -2.0004451274871826, -1.929288387298584),
        (5.463134765625, 5.463134765625, 5.463134765625)],
    [(-2.9547364711761475, -3.17536997795105, -3.143850803375244),
        (5.305514812469482, 4.535006523132324, 3.3618252277374268)],
    [(-1.2906527519226074, -1.2906527519226074, -1.2906527519226074),
        (2.515115737915039, 2.515115737915039, 2.515115737915039)]
]
# max - min, for each channel, of the values above 
range_minmax_l1_perclass = {
    normal_class_: tuple(
        ma - mi 
        for ma, mi in zip(
            min_max_l1[normal_class_][1], 
            min_max_l1[normal_class_][0],
        )
    )
    for normal_class_ in range(NCLASSES)
}

# mean and std of original images per class
mean = [
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
std = [
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

# these sequences are declared here then composed according to the if cases below
lcn_transform = transforms.Lambda(lambda x: local_contrast_normalization(x))

lcn_global_normalization_transform_perclasss = {
    normal_class_: transforms.Normalize(
        min_max_l1[normal_class_][0], 
        range_minmax_l1_perclass[normal_class_]
    )
    for normal_class_ in range(NCLASSES)
}

normalize_mean_std_perclass = {
    normal_class_: transforms.Normalize(mean[normal_class_], std[normal_class_])
    for normal_class_ in range(NCLASSES)
}

clamp01 = transforms.Lambda(lambda x: x.clamp(0, 1))

img_color_augmentation_sequence = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomChoice([
        transforms.ColorJitter(0.04, 0.04, 0.04, 0.04),
        transforms.ColorJitter(0.005, 0.0005, 0.0005, 0.0005),
    ]),
    transforms.ToTensor(),
    transforms.Lambda(
        lambda x: (
            x 
            + torch.randn_like(x).mul(np.random.randint(0, 2)).mul(0.1 * x.std())
        )
    ),
    clamp01,
])
        
        
class MVTecAnomalyDetectionDataModule(LightningDataModule):

    def __init__(
        self, 
        root: str, 
        normal_class: int, 
        preproc: str, 
        supervise_mode: str, 
        batch_size: int,
        nworkers: int,
        pin_memory: bool,
        raw_shape: Tuple[int, int, int] = (240, 240), 
        net_shape: Tuple[int, int, int] = (224, 224),
        real_anomaly_limit: int = None, 
    ):
        super().__init__()
                
        # validate raw_shape
        assert len(raw_shape) == 2, "raw_shape must be a tuple of length 3"
        
        width, height = raw_shape
        assert width == height, f"raw_shape: width={width} != height={height}"
        del width, height
        
        # validate net_shape
        assert len(net_shape) == 2, "net_shape must be a tuple of length 3"
        
        width, height = net_shape
        assert width == height, f"net_shape: width={width} != height={height}"
        del width, height
                
        # validate preproc
        assert preproc in PREPROCESSING_CHOICES, f'`preproc` must be one of {PREPROCESSING_CHOICES}'
        assert supervise_mode in SUPERVISE_MODES, f'`supervise_mode` must be one of {SUPERVISE_MODES}'

        if supervise_mode == SUPERVISE_MODE_REAL_ANOMALY:
            assert real_anomaly_limit is not None, f"'real_anomaly_limit' cannot be None if the supervise mode is {supervise_mode}"
            assert real_anomaly_limit > 0, f"real_anomaly_limit should be > 0, found {real_anomaly_limit}"

        assert nworkers >= 0, f"nworkers must be >= 0, found {nworkers}"
        
        self.root = root
        self.preproc = preproc
        self.supervise_mode = supervise_mode
        self.batch_size = batch_size
        self.nworkers = nworkers
        self.pin_memory = pin_memory
        self.raw_shape = raw_shape
        self.net_shape = net_shape
        self.real_anomaly_limit = real_anomaly_limit
        
        self.normal_class = normal_class
        self.outlier_classes = list(range(NCLASSES))
        self.outlier_classes.remove(self.normal_class)
        
        img_width_height = self.net_shape[-1]
        
        # different types of preprocessing pipelines, 'lcn' is for using LCN, 'aug{X}' for augmentations
        if preproc == PREPROCESSING_LCNAUG1:
            
            # img and gtmap - train
            # train_img_and_gtmap_transform = transforms.Compose([
            #     # ImgGtmapTensorsToUint8(), 
            #     ImgGtmapToPIL(),
            #     # these apply equally to the img and the gtmap
            #     MultiCompose([
            #         transforms.RandomChoice([
            #             transforms.RandomCrop(img_width_height, padding=0), 
            #             transforms.Resize(img_width_height, Image.NEAREST),
            #         ]),
            #         transforms.ToTensor(),
            #     ]),
            # ])
            train_img_and_gtmap_transform = MultiCompose([
                ImgGtmapToPIL(),
                transforms.RandomChoice([
                    transforms.RandomCrop(img_width_height, padding=0), 
                    transforms.Resize(img_width_height, Image.NEAREST),
                ]),
                transforms.ToTensor(),
            ])
            
            # img - train
            train_img_transform = transforms.Compose([
                img_color_augmentation_sequence,
                lcn_transform, 
                lcn_global_normalization_transform_perclasss[normal_class],
                clamp01,
            ])
            
            # image and gtmap - test
            # test_img_and_gtmap_transform = transforms.Compose([
            #     ImgGtmapTensorsToUint8(), 
            #     ImgGtmapToPIL(),
            #     # use this one or 
            #     # these apply equally to the img and the gtmap
            #     MultiCompose([
            #         transforms.Resize(img_width_height, Image.NEAREST), 
            #         transforms.ToTensor()
            #     ]),
            # ])
            test_img_and_gtmap_transform = MultiCompose([
                ImgGtmapTensorsToUint8(), 
                ImgGtmapToPIL(),
                transforms.Resize(img_width_height, Image.NEAREST), 
                transforms.ToTensor()
            ])
            
            # img - test
            test_img_transform = transforms.Compose([
                lcn_transform, 
                lcn_global_normalization_transform_perclasss[normal_class],
                clamp01,
            ])
            
        else:
            raise ValueError(f'Preprocessing pipeline `{preproc}` is not known.')

        self._online_instance_replacer = OnlineInstanceReplacer(supervise_mode)
        online_supervisor_transform = MultiCompose([
            self._online_instance_replacer,
        ])
                
        self.train_mvtec = MvTec(
            root=self.root, 
            split=SPLIT_TRAIN, 
            img_and_gtmap_transform=train_img_and_gtmap_transform, 
            img_transform=train_img_transform, 
            all_transform=online_supervisor_transform,
            shape=self.raw_shape, 
            normal_class=self.normal_class,
        )
        
        self.test_mvtec = MvTec(
            root=self.root, 
            split=SPLIT_TEST, 
            img_and_gtmap_transform=test_img_and_gtmap_transform, 
            img_transform=test_img_transform, 
            all_transform=None,
            shape=self.raw_shape, 
            normal_class=self.normal_class,
        )

    def prepare_data(self):
        self.train_mvtec.prepare_data()
        self.test_mvtec.prepare_data()

    def setup(self):
        
        self.train_mvtec.setup()
        self.test_mvtec.setup()
        
        if self.supervise_mode == SUPERVISE_MODE_REAL_ANOMALY:
            
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
                supervision_indices = np.random.choice(
                    all_indices, size=min(self.real_anomaly_limit, all_indices.numel()), replace=False,
                )
                test_indices = np.array(sorted(set(all_indices) - set(supervision_indices)))

                self.test_subsplits_indices_per_anomalytype[idx] = dict(
                    supervision=supervision_indices,
                    test=test_indices,
                )
            
            self.test_subsplits_indices = dict(
                supervision=np.sort(np.concatenate(tuple(
                    ss["supervision"] 
                    for ss in self.test_subsplits_indices_per_anomalytype.values()
                ))),
                test=np.sort(np.concatenate(tuple(
                    ss["test"] for ss in self.test_subsplits_indices_per_anomalytype.values()
                ))),
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
            self.real_anomaly_mvtec = Subset(
                self.real_anomaly_mvtec, 
                self.test_subsplits_indices["supervision"],
            )
            self.test_mvtec = Subset(self.test_mvtec, self.test_subsplits_indices["test"])
            
            self._online_instance_replacer.real_anomaly_dataloader = DataLoader(
                dataset=self.real_anomaly_mvtec, 
                batch_size=1,
                shuffle=True,
                num_workers=0, 
                pin_memory=False, 
                persistent_workers=False,
            )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_mvtec, 
            batch_size=self.batch_size, 
            shuffle=True,
            num_workers=self.nworkers, 
            pin_memory=self.pin_memory, 
            persistent_workers=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_mvtec, 
            batch_size=self.batch_size, 
            shuffle=False,
            num_workers=self.nworkers, 
            pin_memory=self.pin_memory, 
            persistent_workers=False,
        )
        

if __name__ == "__main__":
    
    # mvtec = MvTec(
    #     root="../../data/datasets", 
    #     split=SPLIT_TEST, 
    #     # split=SPLIT_TRAIN, 
    #     normal_class=0,
    #     # shape=(3, 100, 100),
    #     shape=(3, 50, 50),
    # )
    # mvtec.prepare_data()
    # mvtec.setup()
    # pass

    mvtecad_datamodule = MVTecAnomalyDetectionDataModule(
        root="../../data/datasets",
        normal_class=0,
        preproc=PREPROCESSING_LCNAUG1,
        supervise_mode=SUPERVISE_MODE_SYNTHETIC_ANOMALY_CONFETTI,
        # supervise_mode=SUPERVISE_MODE_REAL_ANOMALY,
        batch_size=128,
        nworkers=0,
        pin_memory=False,
        raw_shape=(240, 240),
        net_shape=(224, 224),
        real_anomaly_limit=1,
    )
    
    mvtecad_datamodule.prepare_data()
    mvtecad_datamodule.setup()
    train_dl = mvtecad_datamodule.train_dataloader()
    test_dl = mvtecad_datamodule.test_dataloader()

    savedir = Path.home() / "tmp"
    savedir.mkdir(exist_ok=True)
    
    # train
    ((normal_imgs, normal_gtmaps), (anomalous_imgs, anomalous_gtmaps)) = generate_dataloader_images(
        train_dl, 
        nimages_perclass=10,
    )
    single_preview_fig = generate_dataloader_preview_single_fig(
        normal_imgs, normal_gtmaps, anomalous_imgs, anomalous_gtmaps,
    )
    
    def savefig(fig, fname, preview_image_size_factor=1):
        # estimated from the nb of pixel bc each one is a byte
        # in KiB
        png_size = fig.get_figheight() * fig.get_figwidth() / 1024. * preview_image_size_factor ** 2
        
        REFERENCE_LIMIT_PNG_SIZE_KiB = 300
        relative_size = png_size / REFERENCE_LIMIT_PNG_SIZE_KiB
        WARNING_RELATIVE_PNG_SIZE = 1.20
        HARD_LIMIT_PNG_SIZE_KiB = 3000
        
        assert png_size < HARD_LIMIT_PNG_SIZE_KiB, f"The PNG size is {png_size} KiB, which is too big. Please reduce the size of the images."
        
        if relative_size > WARNING_RELATIVE_PNG_SIZE:
            # todo make this a warning in wandb 
            print(f"The PNG size is {png_size} KiB, which is too big. Please reduce the size of the images.")
        
        
        fig.savefig(
            fname=fname,
            # this makes sure that the number of pixels in the image is exactly the 
            # same as the number of pixels in the tensors 
            # (other conditions in the functions that generate the images)
            dpi=preview_image_size_factor, 
            pad_inches=0, bbox_inches='tight', 
        )
        
        # import matplotlib.pyplot as plt
        # plt.figure(fig.label)
        # plt.show()
        
    savefig(single_preview_fig, savedir / f"train.{single_preview_fig.label}.png", preview_image_size_factor=1)
    
    # test
    ((normal_imgs, normal_gtmaps), (anomalous_imgs, anomalous_gtmaps)) = generate_dataloader_images(
        test_dl, 
        nimages_perclass=20,
    )
    single_preview_fig = generate_dataloader_preview_single_fig(
        normal_imgs, normal_gtmaps, anomalous_imgs, anomalous_gtmaps,
    )
    savefig(single_preview_fig, savedir / f"test.{single_preview_fig.label}.png", preview_image_size_factor=.4)
    