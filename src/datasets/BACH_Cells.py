import torch
from torch.utils.data import Dataset
import os
from src.utilities.os_utilities import create_dir_if_not_exist
from torchvision.utils import save_image


class BACH_Cells(Dataset):
    def __init__(self, src_folder, img_dim=64, transforms=None):
        super(BACH_Cells, self).__init__()
        self.src_folder = src_folder
        self.img_dim = img_dim

        create_dir_if_not_exist(self.cell_img_dir)
        if(len(os.listdir(self.cell_img_dir)) == 0):
            self.compile_cells()

        self.img_augmentation = transforms

    def compile_cells(self):
        n = 0
        for graph_path in self.graph_paths:
            graph = torch.load(graph_path)
            x = graph.x
            y = graph.y
            for cell_id in range(n, n+x.shape[0]):
                cell = x[cell_id-n].unflatten(3, self.img_dim, self.img_dim)
                torch.save({'img': cell, 'diagnosis': y}, (os.path.join(self.cell_dir, f'{cell_id}.pt')))
                save_image(cell, os.path.join(self.cell_img_dir, f'{cell_id}.png'))
            n += x.shape[0]

    @property
    def cell_img_dir(self):
        return os.path.join(self.src_folder, "CELL_CROPS")

    @property
    def cell_dir(self):
        return os.path.join(self.src_folder, "CELLS")

    @property
    def graph_dir(self):
        return os.path.join(self.src_folder, "GRAPH")

    @property
    def graph_file_names(self):
        return [f for f in os.listdir(self.graph_dir) if ".pt" in f]

    @property
    def graph_paths(self):
        return [os.path.join(self.graph_dir, f) for f in self.graph_file_names]

    @property
    def cell_paths(self):
        return [os.path.join(self.cell_dir, f) for f in os.listdir(self.cell_dir) if f[-2:] == "pt"]

    def __len__(self):
        return len(os.listdir(self.cell_img_dir))

    def __getitem__(self, ind):
        path = self.cell_paths[ind]
        data = torch.load(path)
        cell = data['img']
        pred = data['diagnosis']
        if self.img_augmentation is not None:
            cell = self.graph_augmentation(cell)
        return cell, pred
