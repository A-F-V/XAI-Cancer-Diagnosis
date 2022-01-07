import torch
from torch.utils.data import Dataset
import os
from PIL import Image
import numpy as np
from torchvision.transforms import ToTensor, Compose
from src.transforms.graph_construction.hover_maps import hover_map


class MoNuSeg(Dataset):
    def __init__(self, src_folder=os.path.join("data", "processed", "MoNuSeg_TRAIN"), transform=None):
        """Creates a Dataset object for the MoNuSeg dataset.

        Args:
            src_folder (str optional): The location of the MoNuSeg data set - already processed via the setup script. Defaults to None.
            transform : A transformation that will be performed on both the image and masks. Defaults to None.
            mode (str optional): The mode of the dataset. Can be either "binary_mask" or "instance_mask". Defaults to "binary_mask".
        """
        self.src_folder = src_folder if src_folder else os.path.join(os.getcwd(), 'data', 'processed', 'MoNuSeg_TRAIN')
        self.transform = transform if transform else Compose([])
        self.length = len(os.listdir(os.path.join(self.src_folder, 'images')))

    def __getitem__(self, index):
        img_path = os.path.join(self.src_folder, 'images', f'{index}.tif')
        semantic_mask_path = os.path.join(self.src_folder, 'semantic_masks', f'{index}.npy')
        instance_mask_path = os.path.join(self.src_folder, 'instance_masks', f'{index}.npy')
        item = {"image": ToTensor()(Image.open(img_path)), "semantic_mask": np.load(
            semantic_mask_path, mmap_mode='r+'), "instance_mask": np.load(instance_mask_path, mmap_mode='r+')}
        item['instance_mask'] = torch.as_tensor(item['instance_mask'].astype("int16")).unsqueeze(0)
        item['semantic_mask'] = torch.as_tensor(item['semantic_mask'].astype("int16") > 0).int().unsqueeze(0)

        assert len(item['image'].shape) == 3
        assert len(item['instance_mask'].shape) == 3
        assert len(item['semantic_mask'].shape) == 3
        assert item['semantic_mask'].max() <= 1
        item = self.transform(item)
        item["hover_map"] = hover_map(item["instance_mask"].squeeze())
        return item

    def __len__(self):
        return self.length
