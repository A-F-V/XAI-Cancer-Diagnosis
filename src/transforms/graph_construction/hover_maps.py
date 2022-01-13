from src.utilities.matplot_utilities import BoundingBox
import torch
from torch import Tensor
import numpy as np


def _find_bounding_box(mask: Tensor, nucleus_id):
    nucleus = mask == nucleus_id
    loc = torch.nonzero(nucleus)
    left, right = loc[:, 1].min().item(), loc[:, 1].max().item()
    bottom, top = loc[:, 0].min().item(), loc[:, 0].max().item()
    return BoundingBox(left, top, right, bottom)


def _calculate_median_loc(arr, tot_value):
    count = 0
    for i in range(len(arr)):
        if count+arr[i] >= tot_value/2:
            return i
        count += arr[i]
    raise ("Invalid inputs")


def find_centre_of_mass(img, nucleus_id):
    if not isinstance(img, Tensor):
        img = torch.as_tensor(img)
    mass = (img == nucleus_id)
    horiz = torch.sum(mass, dim=0).tolist()
    vert = torch.sum(mass, dim=1).tolist()
    tot_mass = torch.sum(mass).item()
    x, y = _calculate_median_loc(horiz, tot_mass), _calculate_median_loc(vert, tot_mass)
    return x, y


def _coord_loc(img):
    height, width = img.shape
    horiz, vert = torch.arange(width), torch.arange(height)
    return torch.meshgrid(horiz, vert, indexing="xy")


def hover_map(mask):  # todo write docs
    if not isinstance(mask, Tensor):
        if isinstance(mask, np.ndarray):
            mask = mask.astype("int16")
        mask = torch.as_tensor(mask)
    nuclei = mask.max().item()
    mask = mask.float()
    h_map, v_map = torch.zeros_like(mask), torch.zeros_like(mask)
    x_coord, y_coord = _coord_loc(mask)

    def norm_dist(dist_mat):
        mn, mx = torch.min(dist_mat), torch.max(dist_mat)
        return ((dist_mat-mn)/(max((mx-mn).item(), 1)))*2-1

    for nuc_id in range(1, nuclei+1):
        cell_mask = mask.int() == nuc_id
        centre = find_centre_of_mass(mask.int(), nuc_id)
        x_dist, y_dist = x_coord-centre[0], y_coord-centre[1]
        x_dist, y_dist = cell_mask * x_dist, cell_mask * y_dist
        x_dist, y_dist = norm_dist(x_dist), norm_dist(y_dist)
        x_dist, y_dist = cell_mask * x_dist, cell_mask * y_dist
        h_map += x_dist
        v_map += y_dist
    output = torch.stack([h_map, v_map])
    assert output.min() >= -1 and output.max() <= 1
    return output
