from torch import Tensor
import numpy as np
import torch
"""Utilties for both tensors from PyTorch and numpy."""


def map_value_tensor(tensor, orig_value, new_value):
    """Maps the value of a tensor to a new value.
    Args:
        tensor (Tensor): The tensor to map
        orig_value (float): The original value
        new_value (float): The new value
    Returns:
        The tensor with the mapped value
    """
    return map_values_tensor(tensor, [orig_value], [new_value])


def map_value_numpy_array(narr: np.ndarray, orig_value, new_value):
    return map_values_numpy_array(narr, [orig_value], [new_value])


def map_values_tensor(tensor, orig_values, new_values):
    return Tensor(map_values_numpy_array(tensor.numpy(), orig_values, new_values))


def map_values_numpy_array(narr: np.ndarray, orig_values, new_values):
    copy = narr.copy()
    for orig_value, new_value in zip(orig_values, new_values):
        mask = narr == orig_value
        copy = (copy - mask*orig_value + mask*new_value).astype(narr.dtype)
    return copy


def reset_ids(img: np.ndarray):
    """Resets the order of the ids in an image so that they start from 1.

    Args:
        img (np.ndarray): The image with uint ids.

    Returns:
        np.ndarray: The same image with the ids reset and reordered.
    """
    ids = np.sort(np.unique(img)).tolist()
    new_ids = list(range(len(ids)))
    return map_values_numpy_array(img, ids, new_ids)


def gradient(tensor: Tensor, dim=0):  # todo test
    """Calculates the gradient of a tensor along a certain dimension.
    Args:
        tensor (Tensor): The tensor to calculate the gradient of
        dim (int, optional): The dimension to calculate the gradient of. Defaults to 0.
    Returns:
        The gradient of the tensor.
    """
    # Permute to get dim at start, then shift and add at end, and default last gradient as the second last.
    num_dims = tensor.dim()
    axis_reorder = [i for i in range(0, num_dims)]
    axis_reorder[dim] = 0
    axis_reorder[0] = dim
    permuted = tensor.permute(*axis_reorder)
    shifted = torch.concat([permuted[1:], permuted[-1].unsqueeze(0)], dim=0)
    grad = (shifted - permuted)
    grad[-1] = grad[-2]
    return grad.permute(*axis_reorder)
