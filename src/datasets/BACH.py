
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


# todo refactor to use kwargs instead
class BACH(Dataset):
    def __init__(self, src_folder, ids=None, dmin=100, k=7, window_size=64, downsample=1, min_nodes=10, img_augmentation=None, graph_augmentation=None, pred_mode=False):
        super(BACH, self).__init__()
        self.src_folder = src_folder
        self.ids = ids if ids is not None else list(range(1, 401))
        self.dmin = dmin
        self.k = k
        self.window_size = window_size
        self.downsample = downsample
        self.min_nodes = min_nodes
        self.img_augmentation = img_augmentation
        self.pred_mode = pred_mode
        self.graph_augmentation = graph_augmentation
        create_dir_if_not_exist(self.instance_segmentation_dir, False)
        create_dir_if_not_exist(self.graph_dir, False)
        create_dir_if_not_exist(self.prob_graph_dir, False)
        create_dir_if_not_exist(os.path.join(self.instance_segmentation_dir, "VIZUALISED"), False)

    @property
    def original_image_paths(self):
        paths = []
        for i, folder in enumerate(["Benign", "InSitu", "Invasive", "Normal"]):
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
        prefixes = ["b", "is", "iv", "n"]
        return [f"{prefixes[(ind-1)//100]}{(ind-1)%100+1:03}.pt" for ind in self.ids]

    @property
    def instance_segmentation_paths(self):
        return [os.path.join(self.instance_segmentation_dir, f) for f in self.instance_segmentation_file_names]

    @property
    def prob_graph_dir(self):
        return os.path.join(self.src_folder, "PROB_GRAPH")

    @property
    def prob_graph_file_names(self):
        return [f for f in os.listdir(self.prob_graph_dir) if ".pt" in f]

    @property
    def prob_graph_paths(self):
        return [os.path.join(self.prob_graph_dir, f) for f in self.prob_graph_file_names]

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
                thread = GraphExtractor(path, self.graph_dir, window_size=self.window_size,
                                        k=self.k, dmin=self.dmin, downsample=self.downsample, min_nodes=self.min_nodes, img_trans=self.img_augmentation)
                thread.start()
                threads.append(thread)
            for thread in threads:
                thread.join()

    def generate_prob_graphs(self, model):
        for gn in tqdm(self.graph_file_names, desc="Generating Prob Graphs"):
            graph = torch.load(os.path.join(self.graph_dir, gn))
            graph.x = model.forward_pred(graph.x.unflatten(1, (3, 64, 64)))
            torch.save(graph, os.path.join(self.prob_graph_dir, gn))

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
        graph_id = (self.ids[ind]-1) % len(self.graph_file_names)
        path = self.graph_paths[graph_id] if not self.pred_mode else self.prob_graph_paths[graph_id]
        graph = torch.load(path)
        if self.graph_augmentation is not None:
            graph = self.graph_augmentation(graph)
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
