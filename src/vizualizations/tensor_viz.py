from typing import List
import numpy as np
import matplotlib.pyplot as plt
import torch
from src.utilities.vector_utilities import normalize_vec


def plot_tensor_histogram(tensor, bins=30):  # todo get line going through
    """
    Returns a histogram of the tensor (or numpy)
    """
    f = plt.figure()
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.numpy()
    left, right = tensor.min(), tensor.max()
    hist = np.histogram(tensor.flatten(), bins=bins, range=(left, right))
    plt.plot(np.linspace(left, right, bins), hist[0])
