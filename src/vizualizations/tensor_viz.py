import numpy as np
import matplotlib.pyplot as plt
import torch
from src.utilties.vector_utilities import normalize_vec


def plot_tensor_histogram(tensor, bins=30):  # todo get line going through
    """
    Returns a histogram of the tensor.
    """

    left, right = torch.min(tensor).item(), torch.max(tensor).item()
    hist = np.histogram(tensor.flatten(), bins=bins, range=(left, right))
    plt.scatter(np.linspace(left, right, bins), hist[0])
    plt.show()
