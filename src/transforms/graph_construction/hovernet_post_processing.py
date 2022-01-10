from PIL import ImageFilter
from torchvision.transforms import ToPILImage
from torch.nn.functional import conv2d
from torch import Tensor
from src.transforms.image_processing.filters import sobel
import torch
from skimage.segmentation import watershed
from skimage.feature.peak import peak_local_max
from scipy import ndimage


def _S(hv_maps: Tensor):
    """
    Applies the sobel filter to the hover maps, and gets the importance map from the result.
    Args:
        hv_map (Tensor): hover map of shape (N,2,H,W) and float
    Returns:
        Tensor: The importance map (N,H,W)
    """
    hpx = sobel(hv_maps[:, 0].float())[0].abs()
    hpy = sobel(hv_maps[:, 1].float())[1].abs()
    return torch.maximum(hpx, hpy)


def _markers(q: Tensor, Sm: Tensor, h=0.5, k=0.1):
    """Finds the markers for watershedding.
    Args:
        q (Tensor): The soft semantic map prediction (N,H,W)
        Sm (Tensor): The importance map (N,H,W)
        h (float): The threshold for the semantic map
        k (float): The threshold for the importance map
    Returns:
        Tensor: The markers (N,H,W)
    """
    return torch.maximum(torch.zeros_like(q), (q > h).int()-(Sm > k).int())


def _energy(q: Tensor, Sm: Tensor, h=0.5, k=0.1):
    """Finds the energy map for watershedding.
    Args:
        q (Tensor): The soft semantic map prediction (N,H,W)
        Sm (Tensor): The importance map (N,H,W)
        h (float): The threshold for the semantic map
        k (float): The threshold for the importance map
    Returns:
        Tensor: The energy map (N,H,W)
    """
    return (1-(Sm > k).int())*(q > h).int()


def _watershed(dist: Tensor, mark: Tensor, mask: Tensor = None):
    """Performs watershedding for instance segmentation.

    Args:
        dist (Tensor): The energy landscape (H,W)
        mark (Tensor): The markers (H,W)
        mask (Tensor) optional: A hard mask (H,W)

    Returns:
        Tensor: The prediction (H,W)
    """
    lbs = ndimage.label(mark.numpy())[0]
    return torch.as_tensor(watershed(-(dist.numpy()), markers=lbs, mask=(None if mask is None else mask.numpy()))).int()


# todo do you really want to use the hard mask?
def hovernet_post_process(semantic_mask_pred: Tensor, hv_map_pred: Tensor, h=0.5, k=0.1):  # todo test
    """Takes a prediction and performs instance segmentation. (Usually pre-tiled)

    Args:
        semantic_mask_pred (Tensor): The predicted semantic mask for image (H,W)
        hv_map_pred (Tensor): The predicted hover_maps for the image   (2,H,W)
        h (float): The threshold for the semantic map
        k (float): The threshold for the importance map

    Returns:
        Tensor: The cells instance segmented (H,W) (int)
    """
    sm_hard_pred = (semantic_mask_pred > h).int()
    Sm = _S(hv_map_pred.unsqueeze(0))
    mark = _markers(semantic_mask_pred.unsqueeze(0), Sm, h, k).squeeze()
    energy = _energy(semantic_mask_pred.unsqueeze(0), Sm, h, k).squeeze()
    return _watershed(energy, mark, sm_hard_pred)
