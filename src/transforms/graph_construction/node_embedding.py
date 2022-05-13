from torch import Tensor, nn, as_tensor
from skimage.feature import graycomatrix
from src.utilities.img_utilities import tensor_to_numpy
import numpy as np
import torch
from src.transforms.image_processing.filters import to_gray


def generate_node_embeddings(imgs: Tensor, resnet_encoder: nn.Module, num_neighbours: Tensor, cell_types: Tensor, glcm: Tensor):
    """Generates the node embeddings for my cell graph dataset.

    Args:
        imgs (Tensor): B x 3 x 64 x 64 tensor of images.
        resnet_encoder (nn.Module): The resnet encoder to use. Does predictions incrementally.
        num_neighbours (Tensor): An tensor or array of the each node's neighbours.
        cell_types (Tensor): An tensor or array of the each node's cell type.

    Returns:
        Tensor: B x 312
    """
    num_batches = imgs.shape[0]

    # average RGB
   # argb = imgs.mean(dim=(2, 3))
   # assert argb.shape == (num_batches, 3)

    # Cell_types
    cell_types_one_hot = nn.functional.one_hot(cell_types.to(torch.int64), num_classes=5)
    assert cell_types_one_hot.shape == (num_batches, 5)

    # GLCM

    assert glcm.shape == (num_batches, 50)

    # num_neighbours
    if(len(num_neighbours.shape) == 1):
        num_neighbours = num_neighbours.unsqueeze(1)
    assert num_neighbours.shape == (num_batches, 1)
    num_neighbours /= 40

    # resnet_encoder
    resnet_encoder.eval()
    with torch.no_grad():
        imgs = imgs.to(resnet_encoder.device)
        resnet_encoded = resnet_encoder.predict(imgs)
        assert resnet_encoded.shape == (num_batches, 256, 16, 16)
        resnet_encoded = resnet_encoded.mean(dim=(2, 3))
        assert resnet_encoded.shape == (num_batches, 256)
    resnet_encoded = resnet_encoded.to(glcm.device)
    # concatenate

    final = torch.cat((cell_types_one_hot, num_neighbours, resnet_encoded, glcm), dim=1)
    assert final.shape == (num_batches, 312)
    return final
