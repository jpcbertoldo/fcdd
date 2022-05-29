from collections import Counter
from abc import abstractmethod
from typing import Callable, List, Tuple
import matplotlib

import numpy as np
from matplotlib import pyplot as plt
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image
from collections.abc import Sequence

from matplotlib.figure import Figure
from torch.utils.data import DataLoader

ANOMALY_TARGET = 1
NOMINAL_TARGET = 0


class RandomTransforms:
    """Base class for a list of transformations with randomness
    (joao) implementation copied from torchvision.transforms so the class below could use it
    Args:
        transforms (sequence): list of transformations
    """

    def __init__(self, transforms):
        if not isinstance(transforms, Sequence):
            raise TypeError("Argument transforms should be a sequence")
        self.transforms = transforms

    def __call__(self, *args, **kwargs):
        raise NotImplementedError()

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class RandomChoice(RandomTransforms):
    """
    Pick a transformation randomly picked from a list. This transform does not support torchscript.
    Joao: I am re-implementing this to control the seed. Original implementation copied from torchvision.transforms.transforms.
    """
    
    RETURN_MODE_TRANSFORM_FUNCTION = "transform_function"
    RETURN_MODE_TRANSFORMED_IMAGE = "transformed_image"
    
    def __init__(self, transforms, p=None, random_generator=None, mode=RETURN_MODE_TRANSFORM_FUNCTION):
        super().__init__(transforms)
        assert mode in [self.RETURN_MODE_TRANSFORM_FUNCTION, self.RETURN_MODE_TRANSFORMED_IMAGE], f"mode should be one of {[self.RETURN_MODE_TRANSFORM_FUNCTION, self.RETURN_MODE_TRANSFORMED_IMAGE]}"
        if p is not None and not isinstance(p, Sequence):
            raise TypeError("Argument transforms should be a sequence")
        self.p = p
        self.random_generator = random_generator
        self.mode = mode
        
    def __call__(self, *args):
        picked_transform = self.random_generator.choices(self.transforms, weights=self.p)[0]
        if self.mode == self.RETURN_MODE_TRANSFORM_FUNCTION:
            return picked_transform      
        elif self.mode == self.RETURN_MODE_TRANSFORMED_IMAGE:
            return picked_transform(*args)
        else:
            raise ValueError(f"Unknown mode {self.mode}")

    def __repr__(self):
        format_string = super().__repr__()
        format_string += '(p={0})'.format(self.p)
        format_string += f"(random_generator={self.random_generator})"
        return format_string


class MultiCompose(transforms.Compose):
    """
    Like transforms.Compose, but applies all transformations to a multitude of variables, instead of just one.
    More importantly, for random transformations (like RandomCrop), applies the same choice of transformation, i.e.
    e.g. the same crop for all variables.
    """
    
    def __init__(self, *args, random_generator=None, **kwargs):
        assert random_generator is not None, "You need to pass a random generator to MultiCompose"
        super(MultiCompose, self).__init__(*args, **kwargs)
        self.random_generator = random_generator
    
    def __call__(self, imgs: List[torch.Tensor]):
        
        for img in imgs:
            assert isinstance(img, torch.Tensor), f"MultiCompose only works with torch.Tensor, not {type(img)}"
            assert img.ndim() == 4, f"MultiCompose only works with 4D tensors (batch tensors), not {img.ndim()}"
        
        for t in self.transforms:
            imgs = self.__multi_apply(imgs, t)
            
        return imgs

    def __multi_apply(self, imgs: List, t: Callable):
        # if isinstance(t, transforms.RandomCrop):
        #     for idx, img in enumerate(imgs):
        #         if t.padding is not None and t.padding > 0:
        #             img = TF.pad(img, t.padding, t.fill, t.padding_mode) if img is not None else img
        #         if t.pad_if_needed and img.size[0] < t.size[1]:
        #             img = TF.pad(img, (t.size[1] - img.size[0], 0), t.fill, t.padding_mode) if img is not None else img
        #         if t.pad_if_needed and img.size[1] < t.size[0]:
        #             img = TF.pad(img, (0, t.size[0] - img.size[1]), t.fill, t.padding_mode) if img is not None else img
        #         imgs[idx] = img
        #     i, j, h, w = t.get_params(imgs[0], output_size=t.size)
        #     for idx, img in enumerate(imgs):
        #         imgs[idx] = TF.crop(img, i, j, h, w) if img is not None else img
        # elif isinstance(t, transforms.RandomHorizontalFlip):
        #     if self.random_generator.random() > 0.5:
        #         for idx, img in enumerate(imgs):
        #             imgs[idx] = TF.hflip(img)
        # elif isinstance(t, transforms.RandomVerticalFlip):
        #     if self.random_generator.random() > 0.5:
        #         for idx, img in enumerate(imgs):
        #             imgs[idx] = TF.vflip(img)
        
        if isinstance(t, (transforms.Resize, transforms.Lambda,)):
            return tuple(t(img) for img in imgs)
                
        elif isinstance(t, transforms.RandomChoice):
            raise Exception(f"torchvision.transforms.RandomChoice should be replaced by the implementation in the data module")
        
        elif isinstance(t, RandomChoice):
            assert t.mode == RandomChoice.RETURN_MODE_TRANSFORM_FUNCTION, f"RandomChoice.mode should be {RandomChoice.RETURN_MODE_TRANSFORM_FUNCTION}"
            imgs = self.__multi_apply(imgs, t())
        
        elif isinstance(t, MultiCompose):
            imgs = t(imgs)
            
        else:
            raise Exception(f"Unknown transform {t}")
        
        return imgs


def generate_dataloader_images(dataloader: DataLoader, nimages_perclass=20) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
    normal_imgs: torch.Tensor, 
    normal_gtmaps: torch.Tensor, 
    anomalous_imgs: torch.Tensor,
    anomalous_gtmaps: torch.Tensor,
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
    
    def do(imgs: torch.Tensor, gtmaps: torch.Tensor, label_prefix: str) -> List[Figure]:
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
    normal_imgs: torch.Tensor, 
    normal_gtmaps: torch.Tensor, 
    anomalous_imgs: torch.Tensor,
    anomalous_gtmaps: torch.Tensor,
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
    
    def do(imgs: torch.Tensor, gtmaps: torch.Tensor) -> List[Figure]:
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