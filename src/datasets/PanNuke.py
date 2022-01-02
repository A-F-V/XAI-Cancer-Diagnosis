from torch.utils.data import Dataset
import os
from PIL import Image
import numpy as np
from src.transforms.pytorch.MoNuSeg import ToTensor
from src.utilities.img_utilities import numpy_to_tensor
from torch import Tensor


class PanNuke(Dataset):
    def __init__(self, src_folder=os.path.join("data", "processed", "PanNuke"), transform=None):
        """Creates a Dataset object for the PanNuke dataset.

        Args:
            src_folder (str optional): The location of the PanNuke data set - already processed via the setup script. Defaults to None.
            transform : A transformation that will be performed on both the image and masks. Defaults to None.
        """
        self.src_folder = src_folder if src_folder else os.path.join(os.getcwd(), 'data', 'processed', 'MoNuSeg')
        self.transform = transform if transform else ToTensor()
        self.img_data = np.load(os.path.join(self.src_folder, 'images.npy'))
        self.mask_data = np.load(os.path.join(self.src_folder, 'masks.npy'))
        self.length = 7901

    def __getitem__(self, index):
        item = {"image": numpy_to_tensor(self.img_data[index]), "instance_mask": Tensor(
            self.mask_data[index]).int().squeeze(), "semantic_mask": (Tensor(self.mask_data[index]) != 0).int().squeeze()}
        item = self.transform(item)
        return item

    def __len__(self):
        return self.length
