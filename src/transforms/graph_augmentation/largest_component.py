
import torch.nn as nn
import torch
from torch_geometric.transforms import BaseTransform
import networkx as nx
from torch_geometric.utils import to_networkx, from_networkx


class LargestComponent(BaseTransform):
    def __init__(self):  # set p-mass not to use p
        super().__init__()

    def forward(self, graph):
        G = to_networkx(graph, to_undirected=True, node_attrs=['x', 'pos'], edge_attrs=['edge_attr'])
        Gp = G.subgraph(max(nx.connected_components(G), key=len))
        output = from_networkx(Gp, group_node_attrs=['x', 'pos'], group_edge_attrs=['edge_attr'])
        output.pos = output.x[:, -2:]
        output.x = output.x[:, :-2]
        return output

    def __call__(self, data):
        return self.forward(data)
