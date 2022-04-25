import os
import random
import shutil
import tarfile
import tempfile
import traceback
from abc import ABC, abstractmethod
from collections import Counter
from copy import deepcopy
from itertools import cycle
from typing import Callable, List, Tuple, Union

import numpy as np
import torch
import torchvision.transforms as transforms
from fcdd.datasets.bases import GTMapADDataset, GTMapADDatasetExtension, GTSubset
from fcdd.datasets.preprocessing import ImgGTTargetTransform, MultiCompose
from fcdd.util.logging import Logger
from kornia import gaussian_blur2d
from PIL import Image
from skimage.transform import rotate as im_rotate
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision.datasets import VisionDataset
from torchvision.datasets.imagenet import check_integrity, verify_str_arg
from tqdm import tqdm


# from fcdd.datasets.mvtec_base import MvTec
# from fcdd.datasets.offline_supervisor import \
    # malformed_normal as apply_malformed_normal
# from fcdd.datasets.offline_supervisor import noise as apply_noise
# from fcdd.datasets.noise_modes import generate_noise
# from fcdd.datasets.noise import confetti_noise, smooth_noise


def apply_noise(outlier_classes: List[int], generated_noise: torch.Tensor, norm: torch.Tensor,
          nom_class: int, train_set: Dataset, gt: bool = False) -> Dataset:
    """
    Creates a dataset based on the nominal classes of a given dataset and generated noise anomalies.
    :param outlier_classes: a list of all outlier class indices.
    :param generated_noise: torch tensor of noise images (might also be Outlier Exposure based noise) (n x c x h x w).
    :param norm: torch tensor of nominal images (n x c x h x w).
    :param nom_class: the index of the class that is considered nominal.
    :param train_set: some training dataset.
    :param gt: whether to provide ground-truth maps as well, atm not available!
    :return: a modified dataset, with training data consisting of nominal samples and artificial anomalies.
    """
    if gt:
        raise ValueError('No GT mode for pure noise available!')
    anom = generated_noise.clamp(0, 255).byte()
    data = torch.cat((norm, anom))
    targets = torch.cat(
        (torch.ones(norm.size(0)) * nom_class,
         torch.ones(anom.size(0)) * outlier_classes[0])
    )
    train_set.data = data
    train_set.targets = targets
    return train_set


def apply_malformed_normal(outlier_classes: List[int], generated_noise: torch.Tensor, norm: torch.Tensor, nom_class: int,
                     train_set: Dataset, gt: bool = False, brightness_threshold: float = 0.11*255) -> Dataset:
    """
    Creates a dataset based on the nominal classes of a given dataset and generated noise anomalies.
    Unlike above, the noise images are not directly utilized as anomalies, but added to nominal samples to
    create malformed normal anomalies.
    :param outlier_classes: a list of all outlier class indices.
    :param generated_noise: torch tensor of noise images (might also be Outlier Exposure based noise) (n x c x h x w).
    :param norm: torch tensor of nominal images (n x c x h x w).
    :param nom_class: the index of the class that is considered nominal.
    :param train_set: some training dataset.
    :param gt: whether to provide ground-truth maps as well.
    :param brightness_threshold: if the average brightness (averaged over color channels) of a pixel exceeds this
        threshold, the noise image's pixel value is subtracted instead of added.
        This avoids adding brightness values to bright pixels, where approximately no effect is achieved at all.
    :return: a modified dataset, with training data consisting of nominal samples and artificial anomalies.
    """
    assert (norm.dim() == 4 or norm.dim() == 3) and generated_noise.shape == norm.shape
    norm_dim = norm.dim()
    if norm_dim == 3:
        norm, generated_noise = norm.unsqueeze(1), generated_noise.unsqueeze(1)  # assuming ch dim is skipped
    anom = norm.clone()

    # invert noise for bright regions (bright regions are considered being on average > brightness_threshold)
    generated_noise = generated_noise.int()
    bright_regions = norm.sum(1) > brightness_threshold * norm.shape[1]
    for ch in range(norm.shape[1]):
        gnch = generated_noise[:, ch]
        gnch[bright_regions] = gnch[bright_regions] * -1
        generated_noise[:, ch] = gnch

    anom = (anom.int() + generated_noise).clamp(0, 255).byte()
    data = torch.cat((norm, anom))
    targets = torch.cat(
        (torch.ones(norm.size(0)) * nom_class,
         torch.ones(anom.size(0)) * outlier_classes[0])
    )
    if norm_dim == 3:
        data = data.squeeze(1)
    train_set.data = data
    train_set.targets = targets
    if gt:
        gtmaps = torch.cat(
            (torch.zeros_like(norm)[:, 0].float(),  # 0 for nominal
             (norm != anom).max(1)[0].clone().float())  # 1 for anomalous
        )
        if norm_dim == 4:
            gtmaps = gtmaps.unsqueeze(1)
        return train_set, gtmaps
    else:
        return train_set


def ceil(x: float):
    return int(np.ceil(x))


def floor(x: float):
    return int(np.floor(x))


def confetti_noise(size: torch.Size, p: float = 0.01,
                   blobshaperange: Tuple[Tuple[int, int], Tuple[int, int]] = ((3, 3), (5, 5)),
                   fillval: int = 255, backval: int = 0, ensureblob: bool = True, awgn: float = 0.0,
                   clamp: bool = False, onlysquared: bool = True, rotation: int = 0,
                   colorrange: Tuple[int, int] = None) -> torch.Tensor:
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


def smooth_noise(img: torch.Tensor, ksize: int, std: float, p: float = 1.0, inplace: bool = True) -> torch.Tensor:
    """
    Smoothens (blurs) the given noise images with a Gaussian kernel.
    :param img: torch tensor (n x c x h x w).
    :param ksize: the kernel size used for the Gaussian kernel.
    :param std: the standard deviation used for the Gaussian kernel.
    :param p: the chance smoothen an image, on average smoothens p * n images.
    :param inplace: whether to apply the operation inplace.
    """
    if not inplace:
        img = img.clone()
    ksize = ksize if ksize % 2 == 1 else ksize - 1
    picks = torch.from_numpy(np.random.binomial(1, p, size=img.size(0))).bool()
    if picks.sum() > 0:
        img[picks] = gaussian_blur2d(img[picks].float(), (ksize, ) * 2, (std, ) * 2).int()
    return img


def generate_noise(
    noise_mode: str, 
    size: torch.Size, 
    oe_limit: int,
    logger: Logger = None, 
    datadir: str = None
) -> torch.Tensor:
    """
    Given a noise_mode, generates noise images.
    :param noise_mode: one of the available noise_nodes, see MODES:
        'gaussian': choose pixel values based on Gaussian distribution.
        'uniform: choose pixel values based on uniform distribution.
        'blob': images of randomly many white rectangles of random size.
        'mixed_blob': images of randomly many rectangles of random size, approximately half of them
            are white and the others have random colors per pixel.
        'solid': images of solid colors, i.e. one random color per image.
        'confetti': confetti noise as seen in the paper. Random size, orientation, and number.
            They are also smoothened. Half of them are white, the rest is of one random color per rectangle.
        'imagenet': Outlier Exposure with ImageNet.
        'imagenet22k': Outlier Exposure with ImageNet22k, i.e. the full release fall 2011.
        'cifar100': Outlier Exposure with Cifar-100.
        'emnist': Outlier Exposure with EMNIST.
    :param size: number and size of the images (n x c x h x w)
    :param oe_limit: limits the number of different samples for Outlier Exposure
    :param logger: some logger
    :param datadir: the root directory of datsets (for Outlier Exposure)
    :return: a torch tensor of noise images
    """
    if noise_mode is not None:
        if noise_mode in ['confetti']:
            generated_noise_rgb = confetti_noise(
                size, 0.000018, ((8, 8), (54, 54)), fillval=255, clamp=False, awgn=0, rotation=45, colorrange=(-256, 0)
            )
            generated_noise = confetti_noise(
                size, 0.000012, ((8, 8), (54, 54)), fillval=-255, clamp=False, awgn=0, rotation=45
            )
            generated_noise = generated_noise_rgb + generated_noise
            generated_noise = smooth_noise(generated_noise, 25, 5, 1.0)
        
        elif noise_mode in ['mvtec', 'mvtec_gt']:
            raise NotImplementedError('MVTec-AD and MVTec-AD with ground-truth maps is only available with online supervision.')
        
        else:
            raise NotImplementedError('Supervise noise mode {} unknown (offline version).'.format(noise_mode))
        return generated_noise


def get_target_label_idx(labels: np.ndarray, targets: np.ndarray):
    """
    Get the indices of labels that are included in targets.
    :param labels: array of labels
    :param targets: list/tuple of target labels
    :return: list with indices of target labels
    """
    return np.argwhere(np.isin(labels, targets)).flatten().tolist()


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


class BaseADDataset(ABC):
    """ Anomaly detection dataset base class """

    def __init__(self, root: str, logger: Logger = None):
        super().__init__()
        self.root = root  # root path to data

        self.n_classes = 2  # 0: normal, 1: outlier
        self.normal_classes = None  # tuple with original class labels that define the normal class
        self.outlier_classes = None  # tuple with original class labels that define the outlier class

        self._train_set = None  # must be of type torch.utils.data.Dataset
        self._test_set = None  # must be of type torch.utils.data.Dataset

        self.shape = None  # shape of datapoints, c x h x w
        self.raw_shape = None  # shape of datapoint before preprocessing is applied, c x h x w

        self.logger = logger

    @abstractmethod
    def loaders(self, batch_size: int, shuffle_train=True, shuffle_test=False, num_workers: int = 0) -> Tuple[
            DataLoader, DataLoader]:
        """ Implement data loaders of type torch.utils.data.DataLoader for train_set and test_set. """
        pass

    def __repr__(self):
        return self.__class__.__name__

    def logprint(self, s: str, fps: bool = False):
        """ prints a string via the logger """
        if self.logger is not None:
            self.logger.print(s, fps)
        else:
            print(s)


class TorchvisionDataset(BaseADDataset):
    """ TorchvisionDataset class for datasets already implemented in torchvision.datasets """

    @property
    def train_set(self):
        return self._train_set

    @property
    def test_set(self):
        return self._test_set

    def __init__(self, root: str, logger=None):
        super().__init__(root, logger=logger)

    def loaders(self, batch_size: int, shuffle_train=True, shuffle_test=False, num_workers: int = 0)\
            -> Tuple[DataLoader, DataLoader]:
        assert not shuffle_test, \
            'using shuffled test raises problems with original GT maps for GT datasets, thus disabled atm!'
        # classes = None means all classes
        # TODO: persistent_workers=True makes training significantly faster,
        #  but rn this sometimes yields "OSError: [Errno 12] Cannot allocate memory" as there seems to be a memory leak
        train_loader = DataLoader(dataset=self.train_set, batch_size=batch_size, shuffle=shuffle_train,
                                  num_workers=num_workers, pin_memory=False, persistent_workers=False)
        test_loader = DataLoader(dataset=self.test_set, batch_size=batch_size, shuffle=shuffle_test,
                                 num_workers=num_workers, pin_memory=False, persistent_workers=False)
        return train_loader, test_loader

    def preview(self, percls=20, train=True) -> torch.Tensor:
        """
        Generates a preview of the dataset, i.e. it generates an image of some randomly chosen outputs
        of the dataloader, including ground-truth maps if available.
        The data samples already have been augmented by the preprocessing pipeline.
        This method is useful to have an overview of how the preprocessed samples look like and especially
        to have an early look at the artificial anomalies.
        :param percls: how many samples are shown per class, i.e. for anomalies and nominal samples each
        :param train: whether to show training samples or test samples
        :return: a Tensor of images (n x c x h x w)
        """
        self.logprint('Generating dataset preview...')
        # assert num_workers>0, otherwise the OnlineSupervisor is initialized with the same shuffling in later workers
        if train:
            loader, _ = self.loaders(10, num_workers=1, shuffle_train=True)
        else:
            _, loader = self.loaders(10, num_workers=1, shuffle_test=True)
        x, y, gts, out = torch.FloatTensor(), torch.LongTensor(), torch.FloatTensor(), []
        for xb, yb, gtsb in loader:
            x, y, gts = torch.cat([x, xb]), torch.cat([y, yb]), torch.cat([gts, gtsb])
            if all([x[y == c].size(0) >= percls for c in [0, 1]]):
                break
        for c in sorted(set(y.tolist())):
            out.append(x[y == c][:percls])
        if len(gts) > 0:
            assert len(set(gts.reshape(-1).tolist())) <= 2, 'training process assumes zero-one gtmaps'
            out.append(torch.zeros_like(x[y == 0][:percls]))
            for c in sorted(set(y.tolist())):
                g = gts[y == c][:percls]
                if x.shape[1] > 1:
                    g = g.repeat(1, x.shape[1], 1, 1)
                out.append(g)
        self.logprint('Dataset preview generated.')
        return torch.stack([o[:min(Counter(y.tolist()).values())] for o in out])

    def _generate_artificial_anomalies_train_set(self, supervise_mode: str, noise_mode: str, oe_limit: int,
                                                 train_set: Dataset, nom_class: int):
        """
        This method generates offline artificial anomalies,
        i.e. it generates them once at the start of the training and adds them to the training set.
        It creates a balanced dataset, thus sampling as many anomalies as there are nominal samples.
        This is way faster than online generation, but lacks diversity (hence usually weaker performance).
        :param supervise_mode: the type of generated artificial anomalies.
            unsupervised: no anomalies, returns a subset of the original dataset containing only nominal samples.
            other: other classes, i.e. all the true anomalies!
            noise: pure noise images (can also be outlier exposure based).
            malformed_normal: add noise to nominal samples to create malformed nominal anomalies.
            malformed_normal_gt: like malformed_normal, but also creates artificial ground-truth maps
                that mark pixels anomalous where the difference between the original nominal sample
                and the malformed one is greater than a low threshold.
        :param noise_mode: the type of noise used, see :mod:`fcdd.datasets.noise_mode`.
        :param oe_limit: the number of different outlier exposure samples used in case of outlier exposure based noise.
        :param train_set: the training set that is to be extended with artificial anomalies.
        :param nom_class: the class considered nominal
        :return:
        """
        
        if isinstance(train_set.targets, torch.Tensor):
            dataset_targets = train_set.targets.clone().data.cpu().numpy()
        
        else:  # e.g. imagenet
            dataset_targets = np.asarray(train_set.targets)
        
        generated_noise = norm = None
        
        if supervise_mode in ['noise']:
            self._train_set = apply_noise(self.outlier_classes, generated_noise, norm, nom_class, train_set)
        
        elif supervise_mode in ['malformed_normal']:
            self._train_set = apply_malformed_normal(self.outlier_classes, generated_noise, norm, nom_class, train_set)
        
        elif supervise_mode in ['malformed_normal_gt']:
            train_set, gtmaps = apply_malformed_normal(
                self.outlier_classes, generated_noise, norm, nom_class, train_set, gt=True
            )
            self._train_set = GTMapADDatasetExtension(train_set, gtmaps)
            
        else:
            raise NotImplementedError('Supervise mode {} unknown.'.format(supervise_mode))
        
    def _generate_noise(self, noise_mode: str, size: torch.Size, oe_limit: int = None, datadir: str = None) -> torch.Tensor:
        generated_noise = generate_noise(noise_mode, size, oe_limit, logger=self.logger, datadir=datadir)
        return generated_noise


class OnlineSupervisor(ImgGTTargetTransform):
    invert_threshold = 0.025

    def __init__(self, ds: TorchvisionDataset, supervise_mode: str, noise_mode: str, oe_limit: int = np.infty,
                 p: float = 0.5, exclude: List[str] = ()):
        """
        This class is used as a Transform parameter for torchvision datasets.
        During training it randomly replaces a sample of the dataset retrieved via the get_item method
        by an artificial anomaly.
        :param ds: some AD dataset for which the OnlineSupervisor is used.
        :param supervise_mode: the type of artificial anomalies to be generated during training.
            See :meth:`fcdd.datasets.bases.TorchvisionDataset._generate_artificial_anomalies_train_set`.
        :param noise_mode: the type of noise used, see :mod:`fcdd.datasets.noise_mode`.
            In addition to the offline noise modes, the OnlineSupervisor offers Outlier Exposure with MVTec-AD.
            The oe_limit parameter for MVTec-AD limits the number of different samples per defection type
            (including "good" instances, i.e. nominal ones in the test set).
        :param oe_limit: the number of different Outlier Exposure samples used in case of outlier exposure based noise.
        :param p: the chance to replace a sample from the original dataset during training.
        :param exclude: all class names that are to be excluded in Outlier Exposure datasets.
        """
        self.ds = ds
        self.supervise_mode = supervise_mode
        self.noise_mode = noise_mode
        self.oe_limit = oe_limit
        self.p = p
        
        self.noise_sampler = None
        
        if noise_mode == 'mvtec':
            self.noise_sampler = cycle(
                OEMvTec(
                    (1, ) + ds.raw_shape, 
                    ds.normal_classes, 
                    limit_var=oe_limit,
                    logger=ds.logger, 
                    root=ds.root,
                ).data_loader()
            )
            
        elif noise_mode == 'mvtec_gt':
            self.noise_sampler = cycle(
                OEMvTec(
                    (1, ) + ds.raw_shape, 
                    ds.normal_classes, 
                    limit_var=oe_limit,
                    logger=ds.logger, 
                    gt=True, 
                    root=ds.root,
                ).data_loader()
            )

    def __call__(self, img: torch.Tensor, gt: torch.Tensor, target: int,
                 replace: bool = None) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Based on the probability defined in __init__, replaces (img, gt, target) with an artificial anomaly.
        :param img: some torch tensor image
        :param gt: some ground-truth map (can be None)
        :param target: some label
        :param replace: whether to force or forbid a replacement, ignoring the probability.
            The probability is only considered if replace == None.
        :return: (img, gt, target)
        """
        if replace or replace is None and random.random() < self.p:
            
            img = img.unsqueeze(0) if img is not None else img
            
            # gt value 1 will be put to anom_label in mvtec_bases get_item
            gt = gt.unsqueeze(0).unsqueeze(0).fill_(1).float() if gt is not None else gt
            
            if self.noise_sampler is None:
                generated_noise = self.ds._generate_noise(
                    self.noise_mode, img.shape
                )
                
            else:
                try:
                    generated_noise = next(self.noise_sampler)
                    
                except RuntimeError:
                    generated_noise = next(self.noise_sampler)
                    self.ds.logger.warning(
                        'Had to resample in online_supervisor __call__ next(self.noise_sampler) because of {}'
                        .format(traceback.format_exc())
                    )
                    
                if isinstance(generated_noise, (tuple, list)):
                    generated_noise, gt = generated_noise
            
            if self.supervise_mode in ['noise']:
                img, gt, target = self.__noise(img, gt, target, self.ds, generated_noise)
            
            elif self.supervise_mode in ['malformed_normal']:
                img, gt, target = self.__malformed_normal(
                    img, gt, target, self.ds, generated_noise, invert_threshold=self.invert_threshold
                )
            
            elif self.supervise_mode in ['malformed_normal_gt']:
                img, gt, target = self.__malformed_normal(
                    img, gt, target, self.ds, generated_noise, use_gt=True,
                    invert_threshold=self.invert_threshold
                )
                
            else:
                raise NotImplementedError('Supervise mode {self.supervise_mode} unknown.')
            
            img = img.squeeze(0) if img is not None else img
            gt = gt.squeeze(0).squeeze(0) if gt is not None else gt
            
        return img, gt, target

    def __noise(self, img: torch.Tensor, gt: torch.Tensor, target: int, ds: TorchvisionDataset,
                generated_noise: torch.Tensor, use_gt: bool = False):
        if use_gt:
            raise ValueError('No GT mode for pure noise available!')
        anom = generated_noise.clamp(0, 255).byte()
        t = 1 if not hasattr(ds, 'anomalous_label') else ds.anomalous_label  # target transform has already been applied
        return anom, gt, t

    def __malformed_normal(self, img: torch.Tensor, gt: torch.Tensor, target: int, ds: TorchvisionDataset,
                           generated_noise: torch.Tensor, use_gt: bool = False, invert_threshold: float = 0.025):
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

        t = 1 if not hasattr(ds, 'anomalous_label') else ds.anomalous_label  # target transform has already been applied

        if use_gt:
            gt = (img != anom).max(1)[0].clone().float()
            gt = gt.unsqueeze(1)  # value 1 will be put to anom_label in mvtec_bases get_item

        return anom, gt, t


class MvTec(VisionDataset, GTMapADDataset):
    """ Implemention of a torch style MVTec dataset """
    url = "ftp://guest:GU%2E205dldo@ftp.softronics.ch/mvtec_anomaly_detection/mvtec_anomaly_detection.tar.xz"
    base_folder = 'mvtec'
    labels = (
        'bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather',
        'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor',
        'wood', 'zipper'
    )
    normal_anomaly_label = 'good'
    normal_anomaly_label_idx = 0

    def __init__(
        self, 
        root: str, 
        split: str = 'train', 
        target_transform: Callable = None,
        img_gt_transform: Callable = None, 
        transform: Callable = None, 
        all_transform: Callable = None,
        shape=(3, 300, 300), 
        normal_classes=(), 
        nominal_label=0, 
        anomalous_label=1,
        logger: Logger = None, 
        enlarge: bool = False,
    ):
        """
        Loads all data from the prepared torch tensors. If such torch tensors containg MVTec data are not found
        in the given root directory, instead downloads the raw data and prepares the tensors.
        They contain labels, images, and ground-truth maps for a fixed size, determined by the shape parameter.
        :param root: directory where the data is to be found.
        :param split: whether to use "train", "test", or "test_anomaly_label_target" data.
            In the latter case the get_item method returns labels indexing the anomalous class rather than
            the object class. That is, instead of returning 0 for "bottle", it returns "1" for "large_broken".
        :param target_transform: function that takes label and transforms it somewhat.
            Target transform is the first transform that is applied.
        :param img_gt_transform: function that takes image and ground-truth map and transforms it somewhat.
            Useful to apply the same augmentation to image and ground-truth map (e.g. cropping), s.t.
            the ground-truth map still matches the image.
            ImgGt transform is the third transform that is applied.
        :param transform: function that takes image and transforms it somewhat.
            Transform is the last transform that is applied.
        :param all_transform: function that takes image, label, and ground-truth map and transforms it somewhat.
            All transform is the second transform that is applied.
        :param download: whether to download if data is not found in root.
        :param shape: the shape (c x h x w) the data should be resized to (images and ground-truth maps).
        :param normal_classes: all the classes that are considered nominal (usually just one).
        :param nominal_label: the label that is to be returned to mark nominal samples.
        :param anomalous_label: the label that is to be returned to mark anomalous samples.
        :param logger: logger
        :param enlarge: whether to enlarge the dataset, i.e. repeat all data samples ten times.
            Consequently, one iteration (epoch) of the data loader returns ten times as many samples.
            This speeds up loading because the MVTec-AD dataset has a poor number of samples and
            PyTorch requires additional work in between epochs.
        """
        super(MvTec, self).__init__(root, transform=transform, target_transform=target_transform)
        self.split = verify_str_arg(split, "split", ("train", "test", "test_anomaly_label_target"))
        self.img_gt_transform = img_gt_transform
        self.all_transform = all_transform
        self.shape = shape
        self.orig_gtmaps = None
        self.normal_classes = normal_classes
        self.nominal_label = nominal_label
        self.anom_label = anomalous_label
        self.logger = logger
        self.enlarge = enlarge

        self.download(shape=self.shape[1:])

        print('Loading dataset from {}...'.format(self.data_file))
        dataset_dict = torch.load(self.data_file)
        self.anomaly_label_strings = dataset_dict['anomaly_label_strings']
        if self.split == 'train':
            self.data, self.targets = dataset_dict['train_data'], dataset_dict['train_labels']
            self.gt, self.anomaly_labels = None, None
        else:
            self.data, self.targets = dataset_dict['test_data'], dataset_dict['test_labels']
            self.gt, self.anomaly_labels = dataset_dict['test_maps'], dataset_dict['test_anomaly_labels']

        if self.enlarge:
            self.data, self.targets = self.data.repeat(10, 1, 1, 1), self.targets.repeat(10)
            self.gt = self.gt.repeat(10, 1, 1) if self.gt is not None else None
            self.anomaly_labels = self.anomaly_labels.repeat(10) if self.anomaly_labels is not None else None
            self.orig_gtmaps = self.orig_gtmaps.repeat(10, 1, 1) if self.orig_gtmaps is not None else None

        if self.nominal_label != 0:
            print('Swapping labels, i.e. anomalies are 0 and nominals are 1, same for GT maps.')
            assert -3 not in [self.nominal_label, self.anom_label]
        print('Dataset complete.')

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int, torch.Tensor]:
        img, label = self.data[index], self.targets[index]

        if self.split == 'test_anomaly_label_target':
            label = self.target_transform(self.anomaly_labels[index])
        if self.target_transform is not None:
            label = self.target_transform(label)

        if self.split == 'train' and self.gt is None:
            assert self.anom_label in [0, 1]
            # gt is assumed to be 1 for anoms always (regardless of the anom_label), since the supervisors work that way
            # later code fixes that (and thus would corrupt it if the correct anom_label is used here in swapped case)
            gtinitlbl = label if self.anom_label == 1 else (1 - label)
            gt = (torch.ones_like(img)[0] * gtinitlbl).mul(255).byte()
        else:
            gt = self.gt[index]

        if self.all_transform is not None:
            img, gt, label = self.all_transform((img, gt, label))
            gt = gt.mul(255).byte() if gt.dtype != torch.uint8 else gt
            img = img.sub(img.min()).div(img.max() - img.min()).mul(255).byte() if img.dtype != torch.uint8 else img

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.transpose(0, 2).transpose(0, 1).numpy(), mode='RGB')
        gt = Image.fromarray(gt.squeeze(0).numpy(), mode='L')

        if self.img_gt_transform is not None:
            img, gt = self.img_gt_transform((img, gt))

        if self.transform is not None:
            img = self.transform(img)

        if self.nominal_label != 0:
            gt[gt == 0] = -3  # -3 is chosen arbitrarily here
            gt[gt == 1] = self.anom_label
            gt[gt == -3] = self.nominal_label

        return img, label, gt

    def __len__(self) -> int:
        return len(self.data)

    def download(self, verbose=True, shape=None, cls=None):
        assert shape is not None or cls is not None, 'original shape requires a class'
        if not check_integrity(self.data_file if shape is not None else self.orig_data_file(cls)):
            tmp_dir = tempfile.mkdtemp()
            self.download_and_extract_archive(
                self.url, os.path.join(self.root, self.base_folder), extract_root=tmp_dir,
            )
            train_data, train_labels = [], []
            test_data, test_labels, test_maps, test_anomaly_labels = [], [], [], []
            anomaly_labels, albl_idmap = [], {self.normal_anomaly_label: self.normal_anomaly_label_idx}

            for lbl_idx, lbl in enumerate(self.labels if cls is None else [self.labels[cls]]):
                if verbose:
                    print('Processing data for label {}...'.format(lbl))
                for anomaly_label in sorted(os.listdir(os.path.join(tmp_dir, lbl, 'test'))):
                    for img_name in sorted(os.listdir(os.path.join(tmp_dir, lbl, 'test', anomaly_label))):
                        with open(os.path.join(tmp_dir, lbl, 'test', anomaly_label, img_name), 'rb') as f:
                            sample = Image.open(f)
                            sample = self.img_to_torch(sample, shape)
                        if anomaly_label != self.normal_anomaly_label:
                            mask_name = self.convert_img_name_to_mask_name(img_name)
                            with open(os.path.join(tmp_dir, lbl, 'ground_truth', anomaly_label, mask_name), 'rb') as f:
                                mask = Image.open(f)
                                mask = self.img_to_torch(mask, shape)
                        else:
                            mask = torch.zeros_like(sample)
                        test_data.append(sample)
                        test_labels.append(cls if cls is not None else lbl_idx)
                        test_maps.append(mask)
                        if anomaly_label not in albl_idmap:
                            albl_idmap[anomaly_label] = len(albl_idmap)
                        test_anomaly_labels.append(albl_idmap[anomaly_label])

                for anomaly_label in sorted(os.listdir(os.path.join(tmp_dir, lbl, 'train'))):
                    for img_name in sorted(os.listdir(os.path.join(tmp_dir, lbl, 'train', anomaly_label))):
                        with open(os.path.join(tmp_dir, lbl, 'train', anomaly_label, img_name), 'rb') as f:
                            sample = Image.open(f)
                            sample = self.img_to_torch(sample, shape)
                        train_data.append(sample)
                        train_labels.append(lbl_idx)

            anomaly_labels = list(zip(*sorted(albl_idmap.items(), key=lambda kv: kv[1])))[0]
            train_data = torch.stack(train_data)
            train_labels = torch.IntTensor(train_labels)
            test_data = torch.stack(test_data)
            test_labels = torch.IntTensor(test_labels)
            test_maps = torch.stack(test_maps)[:, 0, :, :]  # r=g=b -> grayscale
            test_anomaly_labels = torch.IntTensor(test_anomaly_labels)
            torch.save(
                {'train_data': train_data, 'train_labels': train_labels,
                 'test_data': test_data, 'test_labels': test_labels,
                 'test_maps': test_maps, 'test_anomaly_labels': test_anomaly_labels,
                 'anomaly_label_strings': anomaly_labels},
                self.data_file if shape is not None else self.orig_data_file(cls)
            )

            # cleanup temp directory
            for dirpath, dirnames, filenames in os.walk(tmp_dir):
                os.chmod(dirpath, 0o755)
                for filename in filenames:
                    os.chmod(os.path.join(dirpath, filename), 0o755)
            shutil.rmtree(tmp_dir)
        else:
            print('Files already downloaded.')
            return

    def get_original_gtmaps_normal_class(self) -> torch.Tensor:
        """
        Returns ground-truth maps of original size for test samples.
        The class is chosen according to the normal class the dataset was created with.
        This method is usually used for pixel-wise ROC computation.
        """
        assert self.split != 'train', 'original maps are only available for test mode'
        assert len(self.normal_classes) == 1, 'normal classes must be known and there must be exactly one'
        assert self.all_transform is None, 'all_transform would be skipped here'
        assert all([isinstance(t, (transforms.Resize, transforms.ToTensor)) for t in self.img_gt_transform.transforms])
        if self.orig_gtmaps is None:
            self.download(shape=None, cls=self.normal_classes[0])
            orig_ds = torch.load(self.orig_data_file(self.normal_classes[0]))
            self.orig_gtmaps = orig_ds['test_maps'].unsqueeze(1).div(255)
        return self.orig_gtmaps

    @property
    def data_file(self):
        return os.path.join(self.root, self.base_folder, self.filename)

    @property
    def filename(self):
        return "admvtec_{}x{}.pt".format(self.shape[1], self.shape[2])

    def orig_data_file(self, cls):
        return os.path.join(self.root, self.base_folder, self.orig_filename(cls))

    def orig_filename(self, cls):
        return "admvtec_orig_cls{}.pt".format(cls)

    @staticmethod
    def img_to_torch(img, shape=None):
        if shape is not None:
            return torch.nn.functional.interpolate(
                torch.from_numpy(np.array(img.convert('RGB'))).float().transpose(0, 2).transpose(1, 2)[None, :],
                shape
            )[0].byte()
        else:
            return torch.from_numpy(
                np.array(img.convert('RGB'))
            ).float().transpose(0, 2).transpose(1, 2)[None, :][0].byte()

    @staticmethod
    def convert_img_name_to_mask_name(img_name):
        return img_name.replace('.png', '_mask.png')

    @staticmethod
    def download_and_extract_archive(url, download_root, extract_root=None, filename=None, remove_finished=False):
        download_root = os.path.expanduser(download_root)
        if extract_root is None:
            extract_root = download_root
        if not filename:
            filename = os.path.basename(url)
        if not os.path.exists(download_root):
            os.makedirs(download_root)

        MvTec.download_url(url, download_root, filename)

        archive = os.path.join(download_root, filename)
        print("Extracting {} to {}".format(archive, extract_root))
        MvTec.extract_archive(archive, extract_root, remove_finished)

    @staticmethod
    def download_url(url, root, filename=None):
        """Download a file from a url and place it in root.
        Args:
            url (str): URL to download file from
            root (str): Directory to place downloaded file in
            filename (str, optional): Name to save the file under. If None, use the basename of the URL
        """
        from six.moves import urllib

        root = os.path.expanduser(root)
        if not filename:
            filename = os.path.basename(url)
        fpath = os.path.join(root, filename)

        os.makedirs(root, exist_ok=True)

        def gen_bar_updater():
            pbar = tqdm(total=None)

            def bar_update(count, block_size, total_size):
                if pbar.total is None and total_size:
                    pbar.total = total_size
                progress_bytes = count * block_size
                pbar.update(progress_bytes - pbar.n)

            return bar_update

        # check if file is already present locally
        if not check_integrity(fpath, None):
            try:
                print('Downloading ' + url + ' to ' + fpath)
                urllib.request.urlretrieve(url, fpath, reporthook=gen_bar_updater())
            except (urllib.error.URLError, IOError) as e:
                if url[:5] == 'https':
                    url = url.replace('https:', 'http:')
                    print('Failed download. Trying https -> http instead.'
                          ' Downloading ' + url + ' to ' + fpath)
                    urllib.request.urlretrieve(
                        url, fpath,
                        reporthook=gen_bar_updater()
                    )
                else:
                    raise e
            # check integrity of downloaded file
            if not check_integrity(fpath, None):
                raise RuntimeError("File not found or corrupted.")

    @staticmethod
    def extract_archive(from_path, to_path=None, remove_finished=False):
        if to_path is None:
            to_path = os.path.dirname(from_path)

        with tarfile.open(from_path, 'r:xz') as tar:
            tar.extractall(path=to_path)


class OEMvTec(MvTec):
    def __init__(self, size: torch.Size, clsses: List[int], root: str = None, limit_var: int = np.infty,
                 limit_per_anomaly=True, logger: Logger = None, gt=False, remove_nominal=True):
        """
        Outlier Exposure dataset for MVTec-AD. Considers only a part of the classes.
        :param size: size of the samples in n x c x h x w, samples will be resized to h x w. If n is larger than the
            number of samples available in MVTec-AD, dataset will be enlarged by repetitions to fit n.
            This is important as exactly n images are extracted per iteration of the data_loader.
            For online supervision n should be set to 1 because only one sample is extracted at a time.
        :param clsses: the classes that are to be considered, i.e. all other classes are dismissed.
        :param root: root directory where data is found or is to be downloaded to.
        :param limit_var: limits the number of different samples, i.e. randomly chooses limit_var many samples
            from all available ones to be the training data.
        :param limit_per_anomaly: whether limit_var limits the number of different samples per type
            of defection or overall.
        :param download: whether to download the data if it is not found in root.
        :param logger: logger.
        :param gt: whether ground-truth maps are to be included in the data.
        :param remove_nominal: whether nominal samples are to be excluded from the data.
        """
        assert len(size) == 4 and size[2] == size[3]
        assert size[1] in [1, 3]
        self.root = root
        self.logger = logger
        self.size = size
        self.use_gt = gt
        self.clsses = clsses
        super().__init__(root, 'test', shape=size[1:], logger=logger)

        self.img_gt_transform = MultiCompose([
            transforms.Resize((size[2], size[2])),
            transforms.ToTensor()
        ])
        self.picks = get_target_label_idx(self.targets, self.clsses)
        if remove_nominal:
            self.picks = sorted(list(set.intersection(
                set(self.picks),
                set((self.anomaly_labels != self.normal_anomaly_label_idx).nonzero().squeeze().tolist())
            )))
        if limit_per_anomaly and limit_var is not None:
            new_picks = []
            for l in set(self.anomaly_labels.tolist()):
                linclsses = list(set.intersection(
                    set(self.picks), set((self.anomaly_labels == l).nonzero().squeeze().tolist())
                ))
                if len(linclsses) == 0:
                    continue
                if limit_var < len(linclsses):
                    new_picks.extend(np.random.choice(linclsses, size=limit_var, replace=False))
                else:
                    self.logprint(
                        'OEMvTec shall be limited to {} samples per anomaly label, '
                        'but MvTec anomaly label {} contains only {} samples, thus using all.'
                        .format(limit_var, self.anomaly_label_strings[l], len(linclsses)), fps=False
                    )
                    new_picks.extend(linclsses)
            self.picks = sorted(new_picks)
        else:
            if limit_var is not None and limit_var < len(self):
                self.picks = np.random.choice(self.picks, size=limit_var, replace=False)
            if limit_var is not None and limit_var > len(self):
                self.logprint(
                    'OEMvTec shall be limited to {} samples, but MvTec contains only {} samples, thus using all.'
                    .format(limit_var, len(self))
                )
        if len(self) < size[0]:
            raise NotImplementedError()

    def __len__(self) -> int:
        return len(self.picks if self.picks is not None else self.targets)

    def data_loader(self) -> DataLoader:
        return DataLoader(dataset=self, batch_size=self.size[0], shuffle=True, num_workers=0)

    def __getitem__(self, index: int) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        index = self.picks[index] if self.picks is not None else index

        image, label, gt = super().__getitem__(index)
        image, gt = image.mul(255).byte(), gt.mul(255).byte()

        if self.use_gt:
            return image, gt
        else:
            return image

    def logprint(self, s: str, fps=True):
        if self.logger is not None:
            self.logger.print(s, fps=fps)
        else:
            print(s)
            

class ADMvTec(TorchvisionDataset):
    
    classes_labels = [
        'bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather',
        'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor',
        'wood', 'zipper'
    ]
    
    preprocessing_choices = ('lcnaug1', 'none')
    
    def __init__(
        self, 
        root: str, 
        normal_class: int, 
        preproc: str, 
        nominal_label: int,
        supervise_mode: str, 
        noise_mode: str, 
        oe_limit: int, 
        logger: Logger = None, 
        raw_shape: int = 240
    ):
        """
        AD dataset for MVTec-AD. 
        If no MVTec data is found in the root directory,
        the data is downloaded and processed to be stored in torch tensors with appropriate size (defined in raw_shape).
        This speeds up data loading at the start of training.
        
        :param root: root directory where data is found or is to be downloaded to
        :param normal_class: the class considered nominal
        :param preproc: the kind of preprocessing pipeline
        :param nominal_label: the label that marks nominal samples in training. The scores in the heatmaps always
            rate label 1, thus usually the nominal label is 0, s.t. the scores are anomaly scores.
        :param supervise_mode: the type of generated artificial anomalies.
            See :meth:`fcdd.datasets.bases.TorchvisionDataset._generate_artificial_anomalies_train_set`.
        :param noise_mode: the type of noise used, see :mod:`fcdd.datasets.noise_mode`.
        :param oe_limit: limits the number of different anomalies in case of Outlier Exposure (defined in noise_mode)
        :param logger: logger
        :param raw_shape: the height and width of the raw MVTec images before passed through the preprocessing pipeline.
        """
        super().__init__(root, logger=logger)
        
        assert preproc in ADMvTec.preprocessing_choices, f'`preproc` must be one of {ADMvTec.preprocessing_choices}'

        self.n_classes = 2  # 0: normal, 1: outlier
        self.shape = (3, 224, 224)
        self.raw_shape = (3,) + (raw_shape, ) * 2
        self.normal_classes = tuple([normal_class])
        self.outlier_classes = list(range(0, 15))
        self.outlier_classes.remove(normal_class)
        assert nominal_label in [0, 1], 'GT maps are required to be binary! `nominal_label` should be in 0 or 1'
        self.nominal_label = nominal_label
        self.anomalous_label = 1 if self.nominal_label == 0 else 0

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
        range_minmax_l1 = [
            tuple(
                ma - mi 
                for ma, mi in zip(
                    min_max_l1[normal_class_][1], 
                    min_max_l1[normal_class_][0],
                )
            )
            for normal_class_ in range(0, len(min_max_l1))
        ]

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
        
        lcn_global_normalization_transform = transforms.Normalize(
            min_max_l1[normal_class][0], 
            range_minmax_l1[normal_class]
        )
        
        lcn_transform_sequence = [lcn_transform, lcn_global_normalization_transform]
        
        img_color_augmentation_sequence = [
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
                ).clamp(0, 1)
            ),
        ]
        
        # different types of preprocessing pipelines, 'lcn' is for using LCN, 'aug{X}' for augmentations
        if preproc == 'lcnaug1':
        
            # gtmap - train
            img_gt_transform = MultiCompose([
                transforms.RandomChoice([
                    transforms.RandomCrop(self.shape[-1], padding=0), 
                    transforms.Resize(self.shape[-1], Image.NEAREST),
                ]),
                transforms.ToTensor(),
            ])
            
            # gtmap - test
            img_gt_test_transform = MultiCompose([
                transforms.Resize(self.shape[-1], Image.NEAREST), 
                transforms.ToTensor()
            ])
            
            # img - test
            test_transform = transforms.Compose(deepcopy(lcn_transform_sequence))
            
            # img - train
            transform = transforms.Compose([
                *deepcopy(img_color_augmentation_sequence),
                *deepcopy(lcn_transform_sequence),
            ])
            
        elif preproc == 'none':
            assert self.raw_shape == self.shape, 'in case of no augmentation, raw shape needs to fit net input shape'
            # gtmap - train & test
            img_gt_transform = img_gt_test_transform = MultiCompose([
                transforms.ToTensor(),
            ])
            # img - train & test
            test_transform = transform = transforms.Compose([
                transforms.Normalize(mean[normal_class], std[normal_class])
            ])
            
        else:
            raise ValueError(f'Preprocessing pipeline `{preproc}` is not known.')

        target_transform = transforms.Lambda(
            lambda x: self.anomalous_label if x in self.outlier_classes else self.nominal_label
        )
        target_transform_test = transforms.Lambda(
            lambda x: self.anomalous_label if x != MvTec.normal_anomaly_label_idx else self.nominal_label
        )

        online_supervisor_transform = MultiCompose([OnlineSupervisor(self, supervise_mode, noise_mode, oe_limit),])
        
        # train
        train_set = MvTec(
            root=self.root, 
            split='train', 
            target_transform=target_transform,
            img_gt_transform=img_gt_transform, 
            transform=transform, 
            all_transform=online_supervisor_transform,
            shape=self.raw_shape, 
            normal_classes=self.normal_classes,
            nominal_label=self.nominal_label, 
            anomalous_label=self.anomalous_label,
            # enlarge dataset by repeating all data samples ten time, speeds up data loading
            enlarge=True,  
        )
        self._train_set = GTSubset(
            train_set, 
            get_target_label_idx(train_set.targets.clone().data.cpu().numpy(), self.normal_classes)
        )
        
        # test
        test_set = MvTec(
            root=self.root, 
            split='test_anomaly_label_target', 
            target_transform=target_transform_test,
            img_gt_transform=img_gt_test_transform, 
            transform=test_transform, 
            shape=self.raw_shape,
            normal_classes=self.normal_classes,
            nominal_label=self.nominal_label, 
            anomalous_label=self.anomalous_label,
            enlarge=False
        )
        test_idx_normal = get_target_label_idx(
            test_set.targets.clone().data.cpu().numpy(), 
            self.normal_classes
        )
        self._test_set = GTSubset(test_set, test_idx_normal)
        
