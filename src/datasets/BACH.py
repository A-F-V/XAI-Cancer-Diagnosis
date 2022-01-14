
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
from torchvision.transforms import Normalize

#! only generate once!


class BACH(Dataset):
    def __init__(self, src_folder, prc_folder, cell_seg_model, ids=None, dmin=100, k=7, window_size=64,):
        super(BACH, self).__init__()
        self.src_folder = src_folder
        self.ids = ids if ids is not None else list(range(1, 401))
        self.dmin = dmin
        self.k = k
        self.window_size = window_size
        self.prc_folder = prc_folder
        self.cell_seg_model = cell_seg_model
        self.processed = False
        create_dir_if_not_exist(self.prc_folder, False)

    @property
    def raw_paths(self):
        paths = []
        for i, folder in enumerate(["Benign", "InSitu", "Invasive", "Normal"]):
            for name in os.listdir(os.path.join(self.src_folder, folder)):
                if ".tif" in name:
                    if self.ids == None or int(name[-7:-4])+i*100 in self.ids:
                        paths.append(os.path.join(self.src_folder, folder, name))
        return paths

    @property
    def processed_dir(self):
        return self.prc_folder

    @property
    def processed_file_names(self):
        return [f"{cond}{ind-i*100:03}.pt" for ind in self.ids for i, cond in enumerate(["b", "is", "iv", "n"])]

    @property
    def processed_paths(self):
        return [os.path.join(self.processed_dir, f) for f in self.processed_file_names]

    # def process(self):
    #    if self.processed and not self.reprocess:
    #        return
    #    for path in tqdm(self.raw_paths, desc="Creating Graphs from BACH"):
    #        img = Image.open(path)
    #        img = self.img_transform(img) if self.img_transform is not None else ToTensor()(
    #            img)
    #        ins_pred = instance_mask_prediction_hovernet(self.cell_seg_model, img, tile_size=128, pre_normalized=True)
    #        img = cut_img_from_tile(img, tile_size=128)
    #        graph = extract_graph(img, ins_pred, window_size=50)
    #        if self.data_augmenation is not None:
    #            graph = self.data_augmenation(graph)
#
    #        file_name = os.path.basename(path)[:-4] + ".pt"
    #        proc_path = os.path.join(self.processed_dir, file_name)
    #        torch.save(graph, proc_path)
#
    #    self.processed = True

    def __len__(self):
        return len(self.processed_file_names)

    def __getitem__(self, ind):
        path = self.processed_paths[ind]
        data = torch.load(path)
        graph = extract_graph(data['image'], data['instance_mask'],
                              window_size=self.window_size, k=self.k, dmin=self.dmin)
        y = torch.tensor([0, 0, 0, 0])

        label = os.path.basename(path)
        if label[0] == 'b':
            y[0] = 1
        elif label[0:2] == 'is':
            y[1] = 1
        elif label[0:2] == 'iv':
            y[2] = 1
        elif label[0] == 'n':
            y[3] = 1
        graph.y = y
        return graph
