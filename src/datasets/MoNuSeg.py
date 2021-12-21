from torch.utils.data import Dataset
import os
from PIL import Image
import numpy as np
from src.transforms.MoNuSeg import ToTensor


class MoNuSeg(Dataset):
    def __init__(self, src_folder=os.path.join("data", "processed", "MoNuSeg"), transform=None, mode="binary_mask"):
        """Creates a Dataset object for the MoNuSeg dataset.

        Args:
            src_folder (str optional): The location of the MoNuSeg data set - already processed via the setup script. Defaults to None.
            transform : A transformation that will be performed on both the image and masks. Defaults to None.
            mode (str optional): The mode of the dataset. Can be either "binary_mask" or "instance_mask". Defaults to "binary_mask".
        """
        self.src_folder = src_folder if src_folder else os.path.join(os.getcwd(), 'data', 'processed', 'MoNuSeg')
        self.transform = transform if transform else ToTensor()
        self.length = len(os.listdir(os.path.join(self.src_folder, 'images')))
        self.mode = mode

    def __getitem__(self, index):
        img_path = os.path.join(self.src_folder, 'images', f'{index}.tif')
        semantic_mask_path = os.path.join(self.src_folder, 'semantic_masks', f'{index}.tif')
        item = {"image": Image.open(img_path)}
        if self.mode == "binary_mask":
            item["semantic_mask"] = Image.open(semantic_mask_path).convert('1')
        item = self.transform(item)
        return item

    def __len__(self):
        return self.length
