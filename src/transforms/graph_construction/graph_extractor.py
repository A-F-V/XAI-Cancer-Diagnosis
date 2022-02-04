from src.model.graph_construction.graph import Graph
from src.transforms.graph_construction.hover_maps import find_centre_of_mass
from torch_geometric.data import Data
import torch
from torch import Tensor
from torchvision.transforms.functional import resize
from torch_geometric.utils import to_networkx, from_networkx
import networkx.algorithms as nx
# todo remove small islands


def extract_graph(img: Tensor, ins_seg: Tensor, window_size=70, k=6, dmin=150, min_nodes=10, downsample=1, img_trans=None):
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
            edge_attr = torch.cat((edge_attr, torch.tensor(
                [[length], [length]])), dim=0)                 # USE 1/DIST**2

    # Perfrom Feature Extraction - NORMALIZE?
    feature_matrix_x = None
    position_matrix = torch.zeros((0, 2))
    for x, y in centres:
        feature = img[:, y-window_size//2:y+window_size//2, x-window_size//2:x+window_size//2]
        feature = resize(feature, size=((window_size//downsample), (window_size//downsample)))

        # performs a transformation to the image (like passing through a VAE or getting mean pixel)
        feature = img_trans(feature) if img_trans != None else feature
        # print(feature.shape)
        feature = feature.flatten()
        if feature_matrix_x == None:
            feature_matrix_x = torch.zeros(0, feature.shape[0])

        feature_matrix_x = torch.cat((feature_matrix_x, feature.unsqueeze(0)), dim=0)
        position_matrix = torch.cat((position_matrix, torch.tensor([[x, y]])), dim=0)
    output = Data(x=feature_matrix_x, edge_index=edges_index, edge_attr=edge_attr, pos=position_matrix)

    G = to_networkx(output, to_undirected=True, node_attrs=['x', 'pos'], edge_attrs=['edge_attr'])
    Gp = nx.operators.compose_all([G.subgraph(g) for g in nx.components.connected_components(G) if len(g) >= min_nodes])
    output = from_networkx(Gp, group_node_attrs=['x', 'pos'], group_edge_attrs=['edge_attr'])
    output.pos = output.x[:, -2:]
    output.x = output.x[:, :-2]
    return output


def mean_pixel_extraction(img: Tensor):
    pixel = img.mean(dim=(1, 2)).flatten()
    return pixel


def principle_pixels_extraction(img: Tensor):
    mean, mini, maxi = img.mean(dim=(1, 2)).flatten(), img.reshape(
        (3, -1)).min(dim=1)[0].flatten(), img.reshape((3, -1)).max(dim=1)[0].flatten()
    return torch.cat((mean, mini, maxi))


def quantiles_pixel_extraction(img: Tensor):
    return torch.cat([img.reshape((3, -1)).quantile(q=q, dim=1).flatten() for q in [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1]])
