
from tqdm import tqdm
import torch
from torch_geometric.data import InMemoryDataset, download_url
from torch.utils.data import Dataset
import os
from PIL import Image
from torchvision.transforms import ToTensor, Compose
from src.transforms.graph_construction.hovernet_post_processing import cut_img_from_tile, instance_mask_prediction_hovernet
from src.transforms.graph_construction.graph_extractor import extract_graph
from src.utilities.os_utilities import create_dir_if_not_exist

#! only generate once!


class BACH(Dataset):
    def __init__(self, src_folder, prc_folder, cell_seg_model, ids=None, img_transform=None, data_augmentation=None, reprocess=False):
        super(BACH, self).__init__()
        self.src_folder = src_folder
        self.ids = ids if ids is not None else list(range(1, 101))
        self.img_transform = img_transform
        self.data_augmenation = data_augmentation
        self.reprocess = reprocess
        self.prc_folder = prc_folder
        self.cell_seg_model = cell_seg_model
        self.processed = False
        create_dir_if_not_exist(self.prc_folder, False)

    @property
    def raw_paths(self):
        paths = []
        for folder in ["Benign", "InSitu", "Invasive", "Normal"]:
            for name in os.listdir(os.path.join(self.src_folder, folder)):
                if ".tif" in name:
                    if self.ids == None or int(name[-7:-4]) in self.ids:
                        paths.append(os.path.join(self.src_folder, folder, name))
        return paths

    @property
    def processed_dir(self):
        return self.prc_folder

    @property
    def processed_file_names(self):
        return [f"{cond}{ind:03}.pt" for cond in ["b", "is", "iv", "n"] for ind in self.ids]

    @property
    def processed_paths(self):
        return [os.path.join(self.processed_dir, f) for f in self.processed_file_names]

    def process(self):
        if self.processed and not self.reprocess:
            return
        for path in tqdm(self.raw_paths, desc="Creating Graphs from BACH"):
            img = Image.open(path)
            img = self.img_transform(img) if self.img_transform is not None else ToTensor()(
                img)  # IMG ALREADY NORMALIZED!!!
            ins_pred = instance_mask_prediction_hovernet(self.cell_seg_model, img, tile_size=128)
            img = cut_img_from_tile(img, tile_size=128)
            graph = extract_graph(img, ins_pred, window_size=50)
            if self.data_augmenation is not None:
                graph = self.data_augmenation(graph)

            file_name = os.path.basename(path)[:-4] + ".pt"
            proc_path = os.path.join(self.processed_dir, file_name)
            torch.save(graph, proc_path)

        self.processed = True

    def __len__(self):
        return len(self.processed_file_names)

    def __getitem__(self, id):
        path = self.processed_paths[id]
        return torch.load(path)
