

import imp
import math
from re import M
from torch import Tensor
import torch
import numpy as np


def normalize(vec: Tensor):
    return vec / vec.norm(p=2, dim=0)


def get_stain_vectors(img: Tensor, alpha=0.01, beta=0.15):
    """Gets the stain vectors for an image in RGB
        Implemets the Macenko et al. (2016) method for stain normalization.
    Args:
        img (tensor): The RGB H&E image
    """
    flat_img = img.flatten(1, 2)

    # 1) get optical density

    od = -torch.log10(flat_img)

    # 2) prune small intensity (high noise?). Must have at least component above beta
    # todo is this what the paper says?
    indices = ((od > beta).nonzero().permute(1, 0))[1]
    od = od[:, indices]

    # 3) Get SVD
    svd = torch.svd(od)
    v1, v2 = svd.U[0], svd.U[1]

    assert abs(v1.norm(p=2).item()-1) < 1e-5
    assert abs(v2.norm(p=2).item() - 1) < 1e-5

    assert abs(torch.dot(v1, v2).item()) < 1e-5
    # 4&5) Project points on the the plane and normalize
    perp = torch.cross(v1, v2)
    dist = perp @ od
    proj = od - (perp.unsqueeze(1) @ dist.unsqueeze(0))

    proj = normalize(proj)

    assert abs(proj.norm(p=2, dim=0).mean().item() - 1) < 1e-5

    # 6) Get angles

    angles = torch.acos(torch.matmul(v1.T, proj))
    min_ang, max_ang = np.percentile(angles.numpy(), [alpha, 100-alpha])

    # 7) Get the stain vectors

    stain_v1 = v1*math.cos(min_ang)+v2*math.sin(min_ang)
    stain_v2 = v1*math.cos(max_ang)+v2*math.sin(max_ang)

    # Back to RGB
    stain_v1, stain_v2 = torch.pow(-stain_v1, 10), torch.pow(-stain_v2, 10)
    stain_v1, stain_v2 = normalize(stain_v1), normalize(stain_v2)
    return stain_v1, stain_v2
