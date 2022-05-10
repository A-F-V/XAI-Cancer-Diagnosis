
from tqdm import tqdm
import torch
from torch_geometric.data import InMemoryDataset, download_url
from torch.utils.data import Dataset
import os
from PIL import Image
from torchvision.transforms import ToTensor, Compose
from src.transforms.graph_construction.hovernet_post_processing import cut_img_from_tile
from src.transforms.graph_construction.graph_extractor import extract_graph
from src.utilities.os_utilities import create_dir_if_not_exist
from torchvision.transforms import Normalize
from threading import Thread
from torch import Tensor
from torchvision.transforms import Normalize
from src.transforms.image_processing.filters import glcm
from src.transforms.graph_construction.node_embedding import generate_node_embeddings
import re


class GraphExtractor(Thread):
    def __init__(self, instance_seg_path, output_folder, **kwargs):
        super().__init__()
        self.instance_seg_path = instance_seg_path
        self.output_folder = output_folder
        self.kwargs = kwargs

    def run(self):
        path = self.instance_seg_path
        data = torch.load(path)
        # try:
        graph = extract_graph(data['original_image'], data['instance_mask'], data['cell_categories'], **self.kwargs)
        # except:
        #   print(f"Failed to extract anything of value from {path}")
        #    return
        y = torch.tensor([0, 0, 0, 0])

        label = os.path.basename(path)
        if label[0] == 'b':
            y[1] = 1
        elif label[0:2] == 'is':
            y[2] = 1
        elif label[0:2] == 'iv':
            y[3] = 1
        elif label[0] == 'n':
            y[0] = 1
        graph.y = y
        graph_path = os.path.join(self.output_folder, os.path.basename(path))
        if min(graph.x.shape) > 0:
            torch.save(graph, graph_path)
        else:
            print(f"{path} has no nodes")


def _path_to_id(path):
    path = os.path.basename(path)
    assert 7 <= len(path) <= 8
    m = re.match(r'([a-z]+)(\d+)', path)
    offset = int(m.group(2))
    group = ['n', 'b', 'is', 'iv'].index(m.group(1))
    return group * 100 + offset


def _id_to_path(id):
    group = ['n', 'b', 'is', 'iv'][id // 100]
    offset = (id-1) % 100+1
    return f"{group}{offset:03d}.pt"

# todo refactor to use kwargs instead


class BACH(Dataset):
    def __init__(self, src_folder, ids=None, dmin=100, window_size=64, downsample=1, min_nodes=10, img_augmentation=None, graph_augmentation=None, pre_encoded=True):
        super(BACH, self).__init__()
        self.src_folder = src_folder
        self.ids = ids if ids is not None else list(range(1, 401))
        self.ids = list(set(self.ids) & set(map(_path_to_id, self.graph_paths)))
        self.dmin = dmin

        self.window_size = window_size
        self.downsample = downsample
        self.min_nodes = min_nodes
        self.img_augmentation = img_augmentation
        self.graph_augmentation = graph_augmentation

        self.pre_encoded = pre_encoded

        create_dir_if_not_exist(self.instance_segmentation_dir, False)
        create_dir_if_not_exist(self.graph_dir, False)
        create_dir_if_not_exist(self.encoded_graph_dir, False)
        create_dir_if_not_exist(os.path.join(self.instance_segmentation_dir, "VIZUALISED"), False)

    @staticmethod
    def get_train_val_ids(src_folder):
        train_ind, val_ind = [], []
        graph_split = os.path.join(src_folder, "graph_ind.txt")
        with open(graph_split, "r") as f:
            l1 = f.readline().strip()
            l2 = f.readline().strip()
            train_ind = list(map(int, l1[1:-1].split(",")))
            val_ind = list(map(int, l2[1:-1].split(",")))
        return train_ind, val_ind

    @property
    def original_image_paths(self):
        paths = []
        for i, folder in enumerate(["Normal", "Benign", "InSitu", "Invasive"]):
            for name in os.listdir(os.path.join(self.src_folder, folder)):
                if ".tif" in name:
                    if self.ids == None or int(name[-7:-4])+i*100 in self.ids:
                        paths.append(os.path.join(self.src_folder, folder, name))
        return paths

    @property
    def instance_segmentation_dir(self):
        return os.path.join(self.src_folder, "INSTANCE_SEGMENTATION")

    @property
    def instance_segmentation_file_names(self):
        prefixes = ["n", "b", "is", "iv"]
        return [f"{prefixes[(ind-1)//100]}{(ind-1)%100+1:03}.pt" for ind in self.ids]

    @property
    def instance_segmentation_paths(self):
        return [os.path.join(self.instance_segmentation_dir, f) for f in self.instance_segmentation_file_names]

    @property
    def encoded_graph_dir(self):
        return os.path.join(self.src_folder, "ENCODED_GRAPH")

    @property
    def encoded_graph_file_names(self):
        return [f for f in os.listdir(self.encoded_graph_dir) if ".pt" in f]

    @property
    def encoded_graph_paths(self):
        return [os.path.join(self.encoded_graph_dir, f) for f in self.encoded_graph_file_names]

    @property
    def graph_dir(self):
        return os.path.join(self.src_folder, "GRAPH")

    @property
    def graph_file_names(self):
        return [f for f in os.listdir(self.graph_dir) if ".pt" in f]

    @property
    def graph_paths(self):
        return [os.path.join(self.graph_dir, f) for f in self.graph_file_names]

    def generate_graphs(self, num_workers=10):

        for batch in tqdm(range(0, len(self.instance_segmentation_file_names), num_workers)):
            threads = []
            for path in self.instance_segmentation_paths[batch:min(len(self.instance_segmentation_paths), batch+num_workers)]:
                thread = GraphExtractor(path, self.graph_dir, window_size=self.window_size, dmin=self.dmin,
                                        downsample=self.downsample, min_nodes=self.min_nodes, img_trans=self.img_augmentation)
                thread.start()
                threads.append(thread)
            for thread in threads:
                thread.join()

    def generate_encoded_graphs(self, model):
        for gn in tqdm(self.graph_file_names, desc="Generating Encoded Graphs"):
            graph = torch.load(os.path.join(self.graph_dir, gn))
            graph.x = graph.x.unflatten(1, (3, 64, 64))

            glcm_acc = torch.zeros((graph.x.shape[0], 50))
            for i, img in enumerate(graph.x):
                glcm_acc[i] = glcm(img, normalize=True)
            xp = generate_node_embeddings(graph.x, model, graph.num_neighbours, graph.categories, glcm_acc)
            graph.x = xp
            torch.save(graph, os.path.join(self.encoded_graph_dir, gn))

    def generate_node_distribution(self):
        counts = {}
        for path in self.graph_paths:
            graph = torch.load(path)
            neighbours = {i: 0 for i in range(graph.num_nodes)}
            for frm, _ in graph.edge_index.T:
                neighbours[frm.item()] = neighbours.get(frm.item(), 0) + 1
           # print(neighbours)
            for score in neighbours.values():
                counts[score] = counts.get(score, 0) + 1
        output = torch.zeros(len(counts), dtype=torch.int64)
        for i in counts:
            output[i] = counts[i]
        return output

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, ind):
        graph_id = (self.ids[ind])
        path = os.path.join(self.graph_dir if not self.pre_encoded else self.encoded_graph_dir, _id_to_path(graph_id))
        graph = torch.load(path)
        graph.img_id = _id_to_path(graph_id)[:-3]
        if self.graph_augmentation is not None:
            graph = self.graph_augmentation(graph)
        if not self.pre_encoded:

            if self.img_augmentation is not None:
                aug_x = torch.zeros((len(graph.x), 3, 64, 64), dtype=torch.float32)
                for i in range(len(graph.x)):
                    aug_x[i] = self.img_augmentation({'image': graph.x[i].unflatten(0, (3, 64, 64))})[
                        'image']  # unfortunately need to do this
                graph.x = aug_x

            graph.glcm = torch.zeros((graph.x.shape[0], 50))
            for i, img in enumerate(graph.x):
                graph.glcm[i] = glcm(img, normalize=True)
            # graph.x = self.node_embedder(graph)
            # assert graph.x.shape[1] == 315
        graph.y = categorise(graph.y)
        return graph

    def get_graph_seg_pair(self, id):
        f_name = self.instance_segmentation_file_names[id]
        g_path, seg_path = os.path.join(self.graph_dir, f_name), os.path.join(self.instance_segmentation_dir, f_name)
        assert os.path.exists(g_path) and os.path.exists(seg_path), "Graph-Segmentation Pair not found"
        return torch.load(g_path), torch.load(seg_path)


def categorise(t: Tensor):
    if len(t.shape) == 1:
        t = t.unsqueeze(0)
    return t.argmax(dim=1)
