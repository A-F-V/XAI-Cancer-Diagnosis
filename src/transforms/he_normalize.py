import math
from re import M
from torch import Tensor
import torch
import numpy as np
from src.utilities.vector_utilities import normalize_vec, rotate_in_plane, not_neg


def normalize_vec(vec: Tensor):
    return vec / vec.norm(p=2, dim=0)


def get_stain_vectors(img: Tensor, alpha=0.01, beta=0.15, clipping=4):
    """Gets the stain vectors for an image in OD Space
        Implemets the Macenko et al. (2016) method for stain normalization.
    Args:
        img (tensor): The RGB H&E image
    """
    flat_img = img.flatten(1, 2)

    # 1) get optical density
    # - ref Beer Lambert Law
    od = -torch.log10(flat_img)

    # 2) prune small intensity (high noise?). Must have at least component above beta
    indices = ((od > beta).nonzero().permute(1, 0))[1]
    od = od[:, indices]
    od = od.clip(0, clipping)  # black spots are essentially infinitly stained
    # todo get rid of obvious outliers (not just noise but seriously far away spots)
    # todo maybe an online method?
    # 3) Get SVD (actually Eigen decomposition of Covariance)

    covmatrix = torch.cov(od)
    e, v = torch.linalg.eigh(covmatrix)
    v1, v2 = not_neg(normalize_vec(v[:, 2]).float()), not_neg(normalize_vec(v[:, 1]).float())

    assert abs(v1.norm(p=2).item()-1) < 1e-5
    assert abs(v2.norm(p=2).item() - 1) < 1e-5

    assert abs(torch.dot(v1, v2).item()) < 1e-5

    # 4&5) Project points on the the plane and normalize
    perp = torch.cross(v1, v2).float()
    perp = not_neg(normalize_vec(perp))

    dist = perp @ od
    proj = od - (perp.unsqueeze(1) @ dist.unsqueeze(0))
    proj = normalize_vec(proj)  # todo normalize projection or normalize OD? I think projection

    assert abs(proj.norm(p=2, dim=0).mean().item() - 1) < 1e-5

    # 6) Get angles

    # angles = torch.acos(torch.matmul(v1.T, proj).clip(-1, 1))

    # Since some vectors cross v1, cannot use min and max angle of that (as does not distinguish between positive and negative angles)
    # Rotate to fix and then rotate back

    offset_angle = torch.pi/2
    rot_proj = rotate_in_plane(proj, perp, offset_angle)  # in order to make all vectors in the same area
    angles = torch.acos(torch.matmul(v1.T, rot_proj).clip(-1, 1))

    # print(angles.isnan().sum())
    min_ang, max_ang = np.percentile(angles.numpy(), [alpha, 100-alpha])
    min_ang -= offset_angle
    max_ang -= offset_angle
    # print(min_ang,max_ang)
    # print(v1,perp)
    # 7) Get the stain vectors
    stain_v1 = normalize_vec(rotate_in_plane(v1, perp, min_ang))
    stain_v2 = normalize_vec(rotate_in_plane(v1, perp, max_ang))

    assert abs(stain_v1.norm(p=2).item()-1) < 1e-5
    assert abs(stain_v2.norm(p=2).item()-1) < 1e-5

    # Back to RGB
    #stain_v1, stain_v2 = torch.pow(10, -stain_v1), torch.pow(10, -stain_v2)
    #stain_v1, stain_v2 = normalize_vec(stain_v1), normalize_vec(stain_v2)

    if(stain_v1[0] < stain_v2[0]):
        stain_v1, stain_v2 = stain_v2, stain_v1

    return torch.stack([stain_v1, stain_v2])


def normalize_he_image(img: Tensor, alpha=0.01, beta=0.15):  # todo create TESTS
    """Normalizes an H&E image in RGB so that H and E are same as in other experiments
    Args:
        img (tensor): The RGB H&E image
    """
    v1, v2 = get_stain_vectors(img, alpha=0.01)
    standard_v1, standard_v2 = Tensor([0.7247, 0.6274, 0.2849]), Tensor([0.0624, 0.8357, 0.5456])

    old_basis = torch.stack([v1, v2], dim=0).T
    new_basis = torch.stack([standard_v1, standard_v2], dim=0).T

    flat_img = img.flatten(1, 2)
    od = -torch.log10(flat_img)

    new_od = new_basis @ torch.linalg.pinv(old_basis) @ od
    new_od = new_od.unflatten(1, (img.shape[1], img.shape[2]))
    rgb = torch.pow(10, -new_od)
    return rgb
