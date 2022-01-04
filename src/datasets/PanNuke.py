from torch.utils.data import Dataset
import os
from PIL import Image
import numpy as np
from src.transforms.graph_construction.hover_maps import hover_map
from src.transforms.pytorch.MoNuSeg import ToTensor
from src.utilities.img_utilities import numpy_to_tensor
from torch import Tensor
import torch


class PanNuke(Dataset):
    def __init__(self, src_folder=None, transform=None):
        """Creates a Dataset object for the PanNuke dataset.

        Args:
            src_folder (str optional): The location of the PanNuke data set - already processed via the setup script. Defaults to None.
            transform : A transformation that will be performed on both the image and masks. Defaults to None.
        """
        self.src_folder = src_folder if src_folder else os.path.join(os.getcwd(), 'data', 'processed', 'PanNuke')
        print(self.src_folder)
        self.transform = transform if transform else ToTensor()
        self.length = 7901

    def __getitem__(self, index):
        img_data = np.load(os.path.join(self.src_folder, 'images.npy'), mmap_mode='r+')
        mask_data = np.load(os.path.join(self.src_folder, 'masks.npy'), mmap_mode='r+')
        hover_map_data = np.load(os.path.join(self.src_folder, "hover_maps.npy"), mmap_mode='r+')

        img, mask, hv_map = img_data[index].copy(), mask_data[index].copy(), hover_map_data[index].copy()
        item = {"image": numpy_to_tensor(img),
                "instance_mask": torch.as_tensor(mask.astype("int16")).int(),
                "semantic_mask": (torch.as_tensor(mask.astype("int16")) != 0).int()}
        item["hover_map"] = torch.as_tensor(hv_map)
        item = self.transform(item)
        return item

    def __len__(self):
        return self.length
