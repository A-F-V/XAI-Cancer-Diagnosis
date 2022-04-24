from torch import Tensor, nn, as_tensor
from skimage.feature import greycomatrix
from src.utilities.img_utilities import tensor_to_numpy
import numpy as np
import torch
from src.model.architectures.cancer_prediction.cell_encoder import CellEncoder


def generate_node_embeddings(imgs: Tensor, resnet_encoder: nn.Module, num_neighbours: Tensor, cell_types: Tensor):
    """Generates the node embeddings for my cell graph dataset.

    Args:
        imgs (Tensor): B x 3 x 64 x 64 tensor of images.
        resnet_encoder (nn.Module): The resnet encoder to use. Does predictions incrementally.
        num_neighbours (Tensor): An tensor or array of the each node's neighbours.
        cell_types (Tensor): An tensor or array of the each node's cell type.

    Returns:
        Tensor: B x 315
    """
    num_batches = imgs.shape[0]

    # average RGB
    argb = imgs.mean(dim=(2, 3))
    assert argb.shape == (num_batches, 3)

    # Cell_types
    cell_types_one_hot = nn.functional.one_hot(cell_types, num_classes=5)
    assert cell_types_one_hot.shape == (num_batches, 5)

    # GLCM
    glcm = torch.zeros(0, 50)
    for img in imgs:
        cur_glcm = as_tensor(greycomatrix(image=tensor_to_numpy(img), distances=[
                             1], angles=[0, np.pi/2], levels=5)).flatten()
        assert cur_glcm.shape == (50,)
        glcm = torch.cat((glcm, cur_glcm), dim=0)

    assert glcm.shape == (num_batches, 50)
    # num_neighbours
    if(num_neighbours.shape == (num_batches)):
        num_neighbours = num_neighbours.unsqueeze(1)
    assert num_neighbours.shape == (num_batches, 1)

    # resnet_encoder
    resnet_encoder.eval()
    with torch.no_grad():
        resnet_encoded = resnet_encoder.predict(imgs)
        assert resnet_encoded.shape == (num_batches, 256, 32, 32)
        resnet_encoded = resnet_encoded.mean(dim=(2, 3))
        assert resnet_encoded.shape == (num_batches, 256)

    # concatenate

    final = torch.cat((argb, cell_types_one_hot, num_neighbours, resnet_encoded, glcm), dim=0)
    assert final.shape == (num_batches, 315)
    return final


def node_embedder(model_location):
    model = CellEncoder.load_from_checkpoint(model_location)
    model.cuda()

    def inner(graph):
        imgs = graph.x
        cell_types = graph.categories
        num_neighbours = graph.num_neighbours

        return generate_node_embeddings(imgs=imgs, resnet_encoder=model, num_neighbours=num_neighbours, cell_types=cell_types)
    return inner
