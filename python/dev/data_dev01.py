from bdb import effective
from collections import Counter
import random
from abc import abstractmethod
from copy import deepcopy
from typing import Callable, List, Tuple

import numpy as np
from matplotlib import pyplot as plt
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image



ANOMALY_TARGET = 1
NOMINAL_TARGET = 0


class MultiTransform(object):
    """ Class to mark a transform operation that expects multiple inputs """
    n = 0  # amount of expected inputs
    def __init__(self) -> None:
        pass


class ImgGtmapLabelTransform(MultiTransform):
    """ Class to mark a transform operation that expects three inputs: (image, ground-truth map, label) """
    n = 3
    
    @abstractmethod
    def __call__(self, img: torch.Tensor, gtmap: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return img, gtmap, target


class ImgGtmapTransform(MultiTransform):
    """ Class to mark a transform operation that expects two inputs: (image, ground-truth map) """
    n = 2
    @abstractmethod
    def __call__(self, img: torch.Tensor, gtmap: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return img, gtmap


class ImgGtmapTensorsToUint8(ImgGtmapTransform):
    """Make sure that the tensors are uint8 and the min/max are 0/255"""
    
    def __call__(self, img: torch.Tensor, gtmap: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        gtmap = (
            gtmap.mul(255).byte() 
            if gtmap.dtype != torch.uint8 else 
            gtmap
        )
        img = (
            img.sub(img.min()).div(img.max() - img.min()).mul(255).byte() if img.dtype != torch.uint8 else 
            img
        )
        return img, gtmap


class ImgGtmapToPIL(ImgGtmapTransform):
    """
    This one has to be called so that mvtec data is consistent with all other datasets to return a PIL Image.
    """
    
    def __call__(self, img: torch.Tensor, gtmap: torch.Tensor) -> Tuple[Image.Image, Image.Image]:
        #  the transpose is putting the axes as height, width, channel
        # img = Image.fromarray(np.transpose(img.numpy(), (1, 2, 0)), mode='RGB')
        # gtmap = Image.fromarray(gtmap.squeeze(0).numpy(), mode='L')           
        return transforms.ToPILImage(mode="RGB")(img), transforms.ToPILImage("L")(gtmap)


class MultiCompose(transforms.Compose):
    """
    Like transforms.Compose, but applies all transformations to a multitude of variables, instead of just one.
    More importantly, for random transformations (like RandomCrop), applies the same choice of transformation, i.e.
    e.g. the same crop for all variables.
    """
    def __call__(self, imgs: List):
        for t in self.transforms:
            imgs = list(imgs)
            imgs = self.__multi_apply(imgs, t)
        return imgs

    def __multi_apply(self, imgs: List, t: Callable):
        if isinstance(t, transforms.RandomCrop):
            for idx, img in enumerate(imgs):
                if t.padding is not None and t.padding > 0:
                    img = TF.pad(img, t.padding, t.fill, t.padding_mode) if img is not None else img
                if t.pad_if_needed and img.size[0] < t.size[1]:
                    img = TF.pad(img, (t.size[1] - img.size[0], 0), t.fill, t.padding_mode) if img is not None else img
                if t.pad_if_needed and img.size[1] < t.size[0]:
                    img = TF.pad(img, (0, t.size[0] - img.size[1]), t.fill, t.padding_mode) if img is not None else img
                imgs[idx] = img
            i, j, h, w = t.get_params(imgs[0], output_size=t.size)
            for idx, img in enumerate(imgs):
                imgs[idx] = TF.crop(img, i, j, h, w) if img is not None else img
        elif isinstance(t, transforms.RandomHorizontalFlip):
            if random.random() > 0.5:
                for idx, img in enumerate(imgs):
                    imgs[idx] = TF.hflip(img)
        elif isinstance(t, transforms.RandomVerticalFlip):
            if random.random() > 0.5:
                for idx, img in enumerate(imgs):
                    imgs[idx] = TF.vflip(img)
        elif isinstance(t, transforms.ToTensor):
            for idx, img in enumerate(imgs):
                imgs[idx] = TF.to_tensor(img) if img is not None else None
        elif isinstance(
                t, (transforms.Resize, transforms.Lambda, transforms.ToPILImage, transforms.ToTensor,)
        ):
            for idx, img in enumerate(imgs):
                imgs[idx] = t(img) if img is not None else None
        elif isinstance(t, MultiTransform):
            assert t.n == len(imgs)
            imgs = t(*imgs)
        elif isinstance(t, transforms.RandomChoice):
            t_picked = random.choice(t.transforms)
            imgs = self.__multi_apply(imgs, t_picked)
        elif isinstance(t, MultiCompose):
            imgs = t(imgs)
        else:
            raise NotImplementedError('There is no multi compose version of {} yet.'.format(t.__class__))
        return imgs


def generate_dataloader_images(dataloader, nimages_perclass=20) -> torch.Tensor:
    """
    Generates a preview of the dataset, i.e. it generates an image of some randomly chosen outputs
    of the dataloader, including ground-truth maps.
    The data samples already have been augmented by the preprocessing pipeline.
    This method is useful to have an overview of how the preprocessed samples look like and especially
    to have an early look at the artificial anomalies.
    :param nimages_perclass: how many samples are shown per class, i.e. for anomalies and nominal samples each
    :return: four Tensors of images of shape (n x c x h x w): ((normal_imgs, normal_gtmaps), (anomalous_imgs, anomalous_gtmaps)) 
    """
    imgs, y, gtmaps = torch.FloatTensor(), torch.LongTensor(), torch.FloatTensor()
    
    for imgb, yb, gtmapb in dataloader:
        
        imgs, y, gtmaps = torch.cat([imgs, imgb]), torch.cat([y, yb]), torch.cat([gtmaps, gtmapb])
        
        # the number of anomalous in a batch can be random, so we have to keep iterating and checking 
        if (y == NOMINAL_TARGET).sum() >= nimages_perclass and (y == ANOMALY_TARGET).sum() >= nimages_perclass:
            break
        
    n_unique_values_in_gts = len(set(gtmaps.reshape(-1).tolist()))
    assert n_unique_values_in_gts <= 2, 'training process assumes zero-one gtmaps'
    
    effective_nimages_perclass = min(Counter(y.tolist()).values())
    if effective_nimages_perclass != nimages_perclass:
        print(f"could not find {nimages_perclass} on each class, only generated {effective_nimages_perclass} of each")
        
    return torch.tensor((
        (
            imgs[y == NOMINAL_TARGET][:effective_nimages_perclass], 
            gtmaps[y == NOMINAL_TARGET][:effective_nimages_perclass],
        ),
        (
            imgs[y == ANOMALY_TARGET][:effective_nimages_perclass], 
            gtmaps[y == ANOMALY_TARGET][:effective_nimages_perclass],
        ),
    ))
    

def generate_dataloader_preview(dataloader, nimages_perclass):
    print('Generating dataset preview...')
    print('Generating images...')
    ((normal_imgs, normal_gtmaps), (anomalous_imgs, anomalous_gtmaps)) = generate_dataloader_images(dataloader, nimages_perclass)
    print('Images generated.')

    assert anomalous_imgs.shape == normal_imgs.shape
    assert anomalous_gtmaps.shape == normal_gtmaps.shape

    assert anomalous_imgs[0].shape == anomalous_gtmaps[0].shape
    assert anomalous_imgs[-2:].shape == anomalous_gtmaps[-2:].shape

    assert normal_imgs[0].shape == normal_gtmaps[0].shape
    assert normal_imgs[-2:].shape == normal_gtmaps[-2:].shape

    effective_nimages_perclass, imgs_nchannels, height, width = normal_imgs
    
    
    
if __name__ == "__main__":
    from mvtec_dataset_dev01 import MVTecAnomalyDetectionDataModule, PREPROCESSING_LCNAUG1, SUPERVISE_MODE_REAL_ANOMALY
    mvtecad_datamodule = MVTecAnomalyDetectionDataModule(
        root="../../data/datasets",
        normal_class=0,
        # preproc=PREPROCESSING_NONE,
        preproc=PREPROCESSING_LCNAUG1,
        # supervise_mode=SUPERVISE_MODE_SYNTHETIC_ANOMALY_CONFETTI,
        supervise_mode=SUPERVISE_MODE_REAL_ANOMALY,
        batch_size=128,
        nworkers=2,
        pin_memory=False,
        raw_shape=(3, 50, 50),
        net_shape=(3, 50, 50),
        real_anomaly_limit=1,
    )
    mvtecad_datamodule.prepare_data()
    mvtecad_datamodule.setup()
    generate_dataloader_preview(mvtecad_datamodule.train_dataloader(), nimages_perclass=5)
    
    # about test: create option to use original gtmaps or not (resized gtmaps)