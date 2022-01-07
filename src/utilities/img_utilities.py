from torch import Tensor
from torchvision.transforms import Compose, ToPILImage, ToTensor
import numpy as np


def tensor_to_numpy(tensor):
    return np.asarray(ToPILImage()(tensor))


def numpy_to_tensor(numpy):
    return ToTensor()(numpy)
