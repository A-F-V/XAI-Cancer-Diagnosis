from src.model.graph_construction.graph import Graph
from src.transforms.graph_construction.hover_maps import find_centre_of_mass
from torch_geometric.data import Data
import torch
from torch import Tensor
from torchvision.transforms.functional import resize


def extract_graph(orig_img: Tensor, ins_seg: Tensor, window_size=70, k=6, dmin=150, model=None, downsample=2):
    nuclei = ins_seg.max()
    centres = []

    def out_of_view(x, y):  # possible off by one error
        return (x <= window_size//2 or y <= window_size//2 or x >= ins_seg.shape[1]-window_size//2 or y >= ins_seg.shape[0]-window_size//2)

    # Check whats visible
    for i in range(1, nuclei+1):
        x, y = find_centre_of_mass(ins_seg, i)
        if(not out_of_view(x, y)):
            centres.append((x, y))

    # Perform KNN search
    def euclidean_dist(p1, p2):
        return ((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)**0.5

    edges_index = torch.zeros((2, 0), dtype=torch.long)
    edge_attr = torch.zeros((0, 1))
    for sr in range(len(centres)):
        closest = []
        for ds in range(len(centres)):
            if(sr != ds and euclidean_dist(centres[sr], centres[ds]) < dmin):
                closest.append((ds, euclidean_dist(centres[sr], centres[ds])))
        closest = sorted(closest, key=lambda x: x[1])[:min(k, len(closest))]
        for to, length in closest:
            edges_index = torch.cat((edges_index, torch.tensor([[sr, to], [to, sr]])), dim=1)
            edge_attr = torch.cat((edge_attr, torch.tensor([[length], [length]])), dim=0)

    # Perfrom Feature Extraction - NORMALIZE?
    feature_matrix_x = torch.zeros((0, (window_size//downsample)**2*3))  # as 3 channels
    position_matrix = torch.zeros((0, 2))
    for x, y in centres:
        feature = orig_img[:, y-window_size//2:y+window_size//2, x-window_size//2:x+window_size//2]
        feature = resize(feature, size=((window_size//downsample), (window_size//downsample))).flatten()
        feature_matrix_x = torch.cat((feature_matrix_x, feature.unsqueeze(0)), dim=0)
        position_matrix = torch.cat((position_matrix, torch.tensor([[x, y]])), dim=0)
    output = Data(x=feature_matrix_x, edge_index=edges_index, edge_attr=edge_attr, pos=position_matrix)
    return output
