import math
from re import M
from torch import Tensor
import torch
import numpy as np
from src.utilities.vector_utilities import normalize_vec, rotate_in_plane, not_neg
from src.utilities.matplot_utilities import *
import matplotlib.pyplot as plt


def normalize_vec(vec: Tensor):
    return vec / vec.norm(p=2, dim=0)


def get_stain_vectors(img: Tensor, alpha=0.01, beta=0.15, clipping=10, debug=False):
    """Gets the stain vectors for an image in OD Space
        Implemets the Macenko et al. (2016) method for stain normalization.
    Args:
        img (tensor): The RGB H&E image

    Returns:
        h,e (tuple): The H&E stain vectors in OD Space
    """
    ##########
    # FUNCTION FOR DEBUGGING
    #######
    datapoints = img.shape[0]*img.shape[1]

    def random_sample(tensor, amt=datapoints):
        return tensor[:, np.random.choice(tensor.shape[1], amt, replace=False)]

    def to_hex(rgb):
        return "#{:02x}{:02x}{:02x}".format(rgb[0], rgb[1], rgb[2])

    def rgb_color(x, y, z):
        points = torch.stack([x, y, z], dim=0)
        points = (points*256).int().clip(max=255).T
        return [to_hex(points) for points in points]

    ##############
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

    # COV MATRIX APPROACH - Incorrect - need to subtract means beforeS
    #covmatrix = torch.cov(od)
    #e, v = torch.linalg.eigh(covmatrix)
    # v1, v2 = not_neg(normalize_vec(v[:,2]).float()), not_neg(normalize_vec(v[:,1]).float()) # because v[:,2] is the largest eigenvalue

    # SVD APPROACH
    U, Sig, Vh = torch.linalg.svd(od.permute(1, 0), full_matrices=False)
    v1, v2 = not_neg(normalize_vec(Vh[0]).float()), not_neg(normalize_vec(Vh[1]).float())

    assert abs(v1.norm(p=2).item()-1) < 1e-5
    assert abs(v2.norm(p=2).item() - 1) < 1e-5

    assert abs(torch.dot(v1, v2).item()) < 1e-5

    if debug:
        od_sample = random_sample(od, min(datapoints, 10000))
        xod, yod, zod = od_sample[0], od_sample[1], od_sample[2]
        x, y, z = torch.pow(10, -xod), torch.pow(10, -yod), torch.pow(10, -zod)
        f = plt.figure(figsize=(5, 5))
        ax = f.add_subplot(1, 1, 1, projection='3d')
        ax.scatter3D(xod.numpy(), yod.numpy(), zod.numpy(), c=rgb_color(x, y, z))
        draw_vectors_3d(ax, torch.stack([v1, v2], dim=0), length=0.4, color='b')
        draw_plane(ax, v1, v2, color='b')
        #draw_annotated_vector_3d(ax,v1,(0,0,0),"Eigenvector 1")
        #draw_annotated_vector_3d(ax,v2,(0,0,0),"Eigenvector 2")
        ax.set_xlim(0)
        ax.set_ylim(0)
        ax.set_zlim(0)
        ax.set_xlabel('OD RED')
        ax.set_ylabel('OD GREEN')
        ax.set_zlabel('OD BLUE')
        plt.show()

    # 4&5) Project points on the the plane and normalize
    perp = torch.cross(v1, v2).float()
    perp = not_neg(normalize_vec(perp))

    #dist = perp @ od
    #proj = od - (perp.unsqueeze(1) @ dist.unsqueeze(0))
    # proj = normalize_vec(proj)  # todo normalize projection or normalize OD? I think projection
    V = torch.stack([v1, v2], dim=0)
    proj = (V @ od)

    proj = normalize_vec(proj)
    assert abs(proj.norm(p=2, dim=0).mean().item() - 1) < 1e-5

    # 6) Get angles
    angles = torch.atan2(proj[1], proj[0])
    # angles = torch.acos(torch.matmul(v1.T, proj).clip(-1, 1))

    # Since some vectors cross v1, cannot use min and max angle of that (as does not distinguish between positive and negative angles)
    # Rotate to fix and then rotate back

    #offset_angle = torch.pi/2
    # rot_proj = rotate_in_plane(proj, perp, offset_angle)  # in order to make all vectors in the same area
    #angles = torch.acos(torch.matmul(v1.T, rot_proj).clip(-1, 1))

    # print(angles.isnan().sum())
    min_ang, max_ang = np.percentile(angles.numpy(), [alpha, 100-alpha])
    # print(min_ang,max_ang)
    # print(v1,perp)

    # 7) Get the stain vectors

    #stain_v1 = normalize_vec(rotate_in_plane(v1, perp, min_ang))
    #stain_v2 = normalize_vec(rotate_in_plane(v1, perp, max_ang))

    stain_v1 = V.T @ torch.Tensor([np.cos(min_ang), np.sin(min_ang)])
    stain_v2 = V.T @ torch.Tensor([np.cos(max_ang), np.sin(max_ang)])

    # TODO DRAW GRAPH 2
    if debug:
        od_sample = random_sample(od, min(datapoints, 10000))
        dist_sample = perp @ od_sample
        proj_sample = od_sample - (perp.unsqueeze(1) @ dist_sample.unsqueeze(0))
        norm_proj_sample = normalize_vec(proj_sample)

        inverse_basis = torch.linalg.pinv(torch.stack([v1, v2]).T)

        components = inverse_basis @ proj_sample
        norm_components = inverse_basis @ norm_proj_sample
        x, y, z = torch.pow(10, -od_sample[0]), torch.pow(10, -od_sample[1]
                                                          ), torch.pow(10, -od_sample[2])  # rgb of the sampled points

        sv1_comp, sv2_comp = inverse_basis@stain_v1, inverse_basis@stain_v2

        f = plt.figure(figsize=(5, 5))
        ax = f.add_subplot(1, 1, 1)
        ax.scatter(components[0].numpy(), components[1].numpy(), c=rgb_color(x, y, z))
        ax.scatter(norm_components[0].numpy(), norm_components[1].numpy(), c="g")
        draw_vector_2d(ax, sv1_comp*2, color="blue")
        draw_vector_2d(ax, sv2_comp*2, color="blue")
        plt.show()

    # TODO DRAW GRAPH 3
    if debug:
        od_sample = random_sample(od, min(datapoints, 10000))
        xod, yod, zod = od_sample[0], od_sample[1], od_sample[2]
        x, y, z = torch.pow(10, -xod), torch.pow(10, -yod), torch.pow(10, -zod)

        dist_sample = perp @ od_sample
        proj_sample = od_sample - (perp.unsqueeze(1) @ dist_sample.unsqueeze(0))
        norm_proj_sample = normalize_vec(proj_sample)

        f = plt.figure(figsize=(5, 5))
        ax = f.add_subplot(1, 1, 1, projection='3d')
        ax.scatter3D(xod.numpy(), yod.numpy(), zod.numpy(), c=rgb_color(x, y, z))
        draw_vectors_3d(ax, torch.stack([v1, v2], dim=0), length=0.4, color='b')
        draw_plane(ax, v1, v2, color='b')

        ax.scatter3D(norm_proj_sample[0].numpy(), norm_proj_sample[1].numpy(), norm_proj_sample[2].numpy(), c='black')
        draw_vectors_3d(ax, torch.stack([stain_v1, stain_v2], dim=0), length=2, color='b')
        ax.set_xlim(0)
        ax.set_ylim(0)
        ax.set_zlim(0)
        ax.set_xlabel('OD RED')
        ax.set_ylabel('OD GREEN')
        ax.set_zlabel('OD BLUE')
        plt.show()

    assert abs(stain_v1.norm(p=2).item()-1) < 1e-5
    assert abs(stain_v2.norm(p=2).item()-1) < 1e-5

    # Back to RGB
    #stain_v1, stain_v2 = torch.pow(10, -stain_v1), torch.pow(10, -stain_v2)
    #stain_v1, stain_v2 = normalize_vec(stain_v1), normalize_vec(stain_v2)

    if(stain_v1[0] < stain_v2[0]):
        stain_v1, stain_v2 = stain_v2, stain_v1

    return torch.stack([stain_v1, stain_v2])


def singularly_stained_image(v1, v2):
    standard_v1, standard_v2 = Tensor([0.5850, 0.7193, 0.3748]), Tensor([0.2065, 0.8423, 0.4978])
    close1 = torch.dot(v1, standard_v1) > torch.dot(v1, standard_v2)
    close2 = torch.dot(v2, standard_v1) > torch.dot(v2, standard_v2)
    if close1 ^ close2:
        return False
    else:
        return True


def normalize_he_image(img: Tensor, alpha=1, beta=0.15):  # todo change algorithm - get closer to blue to stay blue
    """Normalizes an H&E image in RGB so that H and E are same as in other experiments
    Args:
        img (tensor): The RGB H&E image
    """
    v1, v2 = get_stain_vectors(img, alpha=alpha, beta=beta)
    # if abs(v1[0].item()-v2[0].item()) < 0.3 or min(v1[0].item(), v2[0].item()) > 0.3:
    #    print("Singular")
    #    return img  # This is for what I call singularly stained images

    if singularly_stained_image(v1, v2):
        print("Singular")
        return img
    # standard_v1, standard_v2 = Tensor([0.5850, 0.7193, 0.3748]), Tensor([0.2065, 0.8423, 0.4978])  # (#42306C,#9E2550)
    standard_v1, standard_v2 = Tensor([0.7247, 0.6274, 0.2849]), Tensor([0.0624, 0.8357, 0.5456])  # (#303c84, #DC2548)

    old_basis = torch.stack([v1, v2], dim=0).T
    new_basis = torch.stack([standard_v1, standard_v2], dim=0).T

    flat_img = img.flatten(1, 2)
    od = (-torch.log10(flat_img)).clip(0, 100)

    new_od = new_basis @ torch.linalg.pinv(old_basis) @ od
    new_od = new_od.unflatten(1, (img.shape[1], img.shape[2])).clip(0, 10)
    rgb = torch.pow(10, -new_od).clip(0, 1)  # !TODO A BETTER CLIPPING THAN THIS!
    # corrective recolouring
    e = 1/255
    mask = torch.logical_not((rgb[0].ge(1.-e) & rgb[1].le(0.+e) & rgb[2].le(0.+e)) |
                             (rgb[0].le(0.+e) & rgb[1].le(0.+e) & rgb[2].ge(1.-e))).int()
    mask = mask.unsqueeze(0).repeat(3, 1, 1)
    rgb = rgb*mask
    return rgb


def deconvolve_he_image(img: Tensor, alpha=1, beta=0.15):
    """Splits H&E image into two image, one for each of the stains.

    Args:
        img (Tensor): Original Image
    """
    v1, v2 = get_stain_vectors(img, alpha=alpha, beta=beta)

    old_basis = torch.stack([v1, v2], dim=0).T
    h_space = torch.stack([v1, torch.zeros(3)], dim=0).T
    e_space = torch.stack([torch.zeros(3), v2], dim=0).T

    flat_img = img.flatten(1, 2)
    od = (-torch.log10(flat_img)).clip(0, 100)

    h_od = h_space @ torch.linalg.pinv(old_basis) @ od
    e_od = e_space @ torch.linalg.pinv(old_basis) @ od

    h_od = h_od.unflatten(1, (img.shape[1], img.shape[2])).clip(0, 10)
    e_od = e_od.unflatten(1, (img.shape[1], img.shape[2])).clip(0, 10)

    h_rgb = torch.pow(10, -h_od).clip(0, 1)
    e_rgb = torch.pow(10, -e_od).clip(0, 1)

    return h_rgb, e_rgb

# TODO:
# [ ] Fully document code
