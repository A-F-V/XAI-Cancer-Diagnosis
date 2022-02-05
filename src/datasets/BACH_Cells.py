import torch
from torch.utils.data import Dataset
import os
from src.utilities.os_utilities import create_dir_if_not_exist
from torchvision.utils import save_image
from tqdm import tqdm
from multiprocessing import Pool
from src.algorithm.counting_matrix import CountingMatrix


class BACH_Cells(Dataset):
    def __init__(self, src_folder, img_dim=64, transforms=None):
        super(BACH_Cells, self).__init__()
        self.src_folder = src_folder
        self.img_dim = img_dim

        self.compile_cells()
        self.img_augmentation = transforms

    def compile_cells(self):
        self.cm = CountingMatrix(len(self.graph_paths))
        for i, path in tqdm(enumerate(self.graph_paths)):
            data = torch.load(path)
            self.cm.add_many(i, data.x.shape[0])
        self.cm.cumulate()

    @property
    def graph_dir(self):
        return os.path.join(self.src_folder, "GRAPH")

    @property
    def graph_file_names(self):
        return [f for f in os.listdir(self.graph_dir) if ".pt" in f]

    @property
    def graph_paths(self):
        return [os.path.join(self.graph_dir, f) for f in self.graph_file_names]

    def __len__(self):
        return len(self.cm)

    def __getitem__(self, ind):
        (graph_idx, cell_idx) = self.cm[ind]
        graph = torch.load(self.graph_paths[graph_idx])
        cell = graph.x[cell_idx]
        y = graph.y[cell_idx]
        if self.img_augmentation is not None:
            cell = self.img_augmentation(cell)
        return {'img': cell, "diagnosis": y}
