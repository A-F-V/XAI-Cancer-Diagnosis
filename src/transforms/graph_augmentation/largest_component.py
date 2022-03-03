
import torch.nn as nn
import torch
from torch_geometric.transforms import BaseTransform
import networkx as nx
from torch_geometric.utils import to_networkx
import torch_geometric
from collections import defaultdict
from torch_geometric.data import Data

# THIS IS A REALLY SCREWED UP IMPLEMENTATION


def from_networkx_fast(nxG, pyG, group_node_attrs=None):
    r"""From the Pytorch Geometric, but without the time consuming check
    """
    node_ids = torch.tensor(list(nxG.nodes))
    G = nx.convert_node_labels_to_integers(nxG)
    if not nx.is_directed(G):
        G = G.to_directed(as_view=True)  # NEED TO ADD TO PREVENT DEEP COPYING LOTS OF IMAGES

    edges = list(G.edges)
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    node_attr = {i: pyG.__getattribute__(i)[node_ids] for i in group_node_attrs}
    return Data(y=pyG.y, edge_index=edge_index, **node_attr, edge_attr=torch.zeros((len(edges), 0)))


class LargestComponent(BaseTransform):
    def __init__(self):  # FORGETS ABOUT DISTANCE WEIGHTS
        super().__init__()

    def forward(self, graph):  # fast and efficient way of doing

        G = to_networkx(graph, node_attrs=["x", "pos"], to_undirected=True)
        Gp = G.subgraph(max(nx.connected_components(G), key=len))
        output = from_networkx_fast(Gp, graph, group_node_attrs=["x", "pos"])
        return output

    def __call__(self, data):
        return self.forward(data)
