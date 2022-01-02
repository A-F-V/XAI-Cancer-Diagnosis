from torch import Tensor
import numpy as np
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
