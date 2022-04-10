from torch.utils.data import Dataset
import os
from PIL import Image
import numpy as np
from src.transforms.graph_construction.hover_maps import hover_map
from src.transforms.pytorch.MoNuSeg import ToTensor
from src.utilities.img_utilities import numpy_to_tensor
from torch import Tensor
import torch
from src.utilities.tensor_utilties import map_value_numpy_array, map_values_numpy_array
from tqdm import tqdm


def reorder_ids(img):
    ids = np.sort(np.unique(img)).tolist()
    new_ids = list(range(len(ids)))
    return map_values_numpy_array(img, ids, new_ids)


def aggregate_masks(mask):
    # 6 channel image (use first 5 channels)
    output = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.uint16)
    for i in range(5):
        offset = output.max()
        addition = reorder_ids(mask[:, :, i])+offset
        addition = map_value_numpy_array(addition, offset, 0)  # set background to 0
        output += addition
   # assert output.max()<255
    return output.astype(np.uint16)


def collect_images(src_folder):
    f1, f2, f3 = os.path.join(src_folder, "Fold 1", "images", "fold1", "images.npy"), os.path.join(
        src_folder, "Fold 2", "images", "fold2", "images.npy"), os.path.join(src_folder, "Fold 3", "images", "fold3", "images.npy")
    a1 = np.load(f1).astype(np.uint8)
    a2 = np.load(f2).astype(np.uint8)
    a3 = np.load(f3).astype(np.uint8)
    arr = np.concatenate([a3, a2, a1], axis=0)
    return arr


def collect_masks(src_folder):
    f1, f2, f3 = os.path.join(src_folder, "Fold 1", "masks", "fold1", "masks.npy"), os.path.join(
        src_folder, "Fold 2", "masks", "fold2", "masks.npy"), os.path.join(src_folder, "Fold 3", "masks", "fold3", "masks.npy")
    a1 = np.load(f1).astype(np.uint8)
    a2 = np.load(f2).astype(np.uint8)
    a3 = np.load(f3).astype(np.uint8)
    arr = np.concatenate([a3, a2, a1], axis=0)
    print(arr.shape)
    category_mask = (arr[:, :, :, :5] != 0).astype(np.uint8)
    instance_mask = np.array([aggregate_masks(img) for img in tqdm(
        arr, desc="Generating Instance Masks for PanNuke")], dtype=np.uint16)
    instance_mask = np.expand_dims(instance_mask, axis=3)
    final = np.concatenate([category_mask, instance_mask], axis=3)
    return final


# The ordering of the channels from index 0 to 4 is neoplastic, inflammatory, connective tissue, dead and non-neoplastic epithelial.


class PanNuke(Dataset):
    def __init__(self, src_folder=None, transform=None, ids=None):
        """Creates a Dataset object for the PanNuke dataset.

        Args:
            src_folder (str optional): The location of the PanNuke data set - already processed via the setup script. Defaults to None.
            transform : A transformation that will be performed on both the image and masks. Defaults to None.
        """
        self.src_folder = src_folder if src_folder else os.path.join(os.getcwd(), 'data', 'processed', 'PanNuke')
        self.transform = transform if transform else None
        self.dir_length = 7901
        self.ids = ids

    def __getitem__(self, index):  # TODO FIX
        if self.ids is not None:
            index = self.ids[index]
        img_data = np.load(os.path.join(self.src_folder, 'images.npy'), mmap_mode='r+')
        mask_data = np.load(os.path.join(self.src_folder, 'masks.npy'), mmap_mode='r+')

        img, mask = img_data[index].copy(), mask_data[index].copy()
        item = {"image": numpy_to_tensor(img),
                "instance_mask": torch.as_tensor(mask[:, :, -1].astype("int16")).int().unsqueeze(0),
                # can be derived from instance mask so don't bother saving explicitly
                "semantic_mask": (torch.as_tensor(mask[:, :, -1].astype("int16")) != 0).float().unsqueeze(0),
                "category_mask": torch.as_tensor(mask[:, :, :5].astype("int16")).float().permute(2, 0, 1)}
        item["category_mask"] = torch.cat(
            [item["category_mask"], 1-item["semantic_mask"]], dim=0)  # last is background mask
        item['image_original'] = item['image'].clone()
        assert item['semantic_mask'].max() <= 1
        if self.transform:
            item = self.transform(item)
        item["hover_map"] = hover_map(item["instance_mask"].squeeze()).float()
        return item

    def __len__(self):
        return self.dir_length if self.ids is None else len(self.ids)

    @staticmethod
    def prepare(src_folder, dst_folder):
        """Takes the PanNuke unzipped original folder and prepares it for use.

        Args:
            src_folder (str): The location of the PanNuke unzipped original folder.
            dst_folder (str): The location to save the processed PanNuke data set.
        """
        if not os.path.exists(dst_folder):
            os.makedirs(dst_folder)
        np.save(os.path.join(dst_folder, 'images.npy'), collect_images(src_folder))
        np.save(os.path.join(dst_folder, 'masks.npy'), collect_masks(src_folder))
