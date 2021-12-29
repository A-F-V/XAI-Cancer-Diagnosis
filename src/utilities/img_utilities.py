from torch import Tensor
from torchvision.transforms import Compose, ToPILImage
import numpy as np


def tensor_to_numpy(tensor):
    return np.asarray(ToPILImage()(tensor))
