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

from matplotlib.figure import Figure


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


def generate_dataloader_images(dataloader, nimages_perclass=20) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
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
        (
            imgs[y == NOMINAL_TARGET][:effective_nimages_perclass], 
            gtmaps[y == NOMINAL_TARGET][:effective_nimages_perclass],
        ),
        (
            imgs[y == ANOMALY_TARGET][:effective_nimages_perclass], 
            gtmaps[y == ANOMALY_TARGET][:effective_nimages_perclass],
        ),
    )
    
    print('Images generated.')
    return ret
    

def generate_dataloader_preview_multiple_fig(
    normal_imgs: torch.Tensor, 
    normal_gtmaps: torch.Tensor, 
    anomalous_imgs: torch.Tensor,
    anomalous_gtmaps: torch.Tensor,
    dpi=80,
) -> List[Figure]:
    """
    normal_imgs, anomalous_imgs: tensors of shape (n x 3 x h x w)
    normal_gtmaps,anomalous_gtmaps: tensors of shape (n x 1 x h x w)
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
    
    figsize = (height / dpi, width / dpi)
       
    def do(imgs: torch.Tensor, gtmaps: torch.Tensor, label_prefix: str) -> List[Figure]:
        """([n, 3, h, w], [n, 1, h, w]) -> n plt figures"""
        
        prevs = [
            # dim 1 = height ==> vertical concatenation
            # the repeat(3) is there to make the gtmap "RGB" while keeping it gray 
            torch.cat([img, gtmap.repeat(3, 1, 1)], dim=1)
            for img, gtmap in zip(imgs, gtmaps)
        ]
        figs = []
        for idx, prev in enumerate(prevs):
            fig_kw = dict(frameon=False)
            fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi, **fig_kw)
            fig.label = f"{label_prefix}_{idx}"
            ax.imshow(np.transpose(prev.numpy(), (1, 2, 0)))   
            ax.axis("off")
            figs.append(fig)         
        return figs
    
    normal_figs = do(normal_imgs, normal_gtmaps, "normal")
    anomalous_figs = do(anomalous_imgs, anomalous_gtmaps, "anomalous")
    
    print('Dataset preview generated.')
    
    return normal_figs, anomalous_figs


def generate_dataloader_preview_single_fig(
    normal_imgs: torch.Tensor, 
    normal_gtmaps: torch.Tensor, 
    anomalous_imgs: torch.Tensor,
    anomalous_gtmaps: torch.Tensor,
    dpi=80,
) -> List[Figure]:
    """
    normal_imgs, anomalous_imgs: tensors of shape (n x 3 x h x w)
    normal_gtmaps,anomalous_gtmaps: tensors of shape (n x 1 x h x w)
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
    
    # 4 accounts for: normal img, normal gtmap, anomalous img, anomalous gtmap
    figsize = (4 * height / dpi, nimages * width / dpi)
       
    def do(imgs: torch.Tensor, gtmaps: torch.Tensor) -> List[Figure]:
        """([n, 3, h, w], [n, 1, h, w]) -> n plt figures"""
        
        # concatenates img/gtmap vertically
        prevs = [
            # dim 1 = height ==> vertical concatenation
            # the repeat(3) is there to make the gtmap "RGB" while keeping it gray 
            torch.cat([img, gtmap.repeat(3, 1, 1)], dim=1)
            for img, gtmap in zip(imgs, gtmaps)
        ]
        # concatenate the many prevs horizontally
        return torch.cat(prevs, dim=2)
    
    normal_prevs = do(normal_imgs, normal_gtmaps)
    anomalous_prevs = do(anomalous_imgs, anomalous_gtmaps)
    
    # concatenate normal/anomalous previews vertically
    prev = torch.cat([normal_prevs, anomalous_prevs], dim=1)
    
    fig_kw = dict(frameon=False)
    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi, **fig_kw)
    fig.label = f"preview_{nimages:02d}_images"
    ax.imshow(np.transpose(prev.numpy(), (1, 2, 0)))   
    ax.axis("off")
    
    print('Dataset preview generated.')
    
    return fig




# about test: create option to use original gtmaps or not (resized gtmaps)