
import torch.nn as nn
import torch
from torch_geometric.transforms import BaseTransform
import networkx as nx
from torch_geometric.utils import to_networkx, from_networkx


class EdgeDropout(BaseTransform):
    def __init__(self, p=0.001, p_mass=None):  # set p-mass not to use p
        super().__init__()
        self.p_mass = p_mass
        self.p = p

    def forward(self, graph):
        G = to_networkx(graph, to_undirected=True, node_attrs=['x', 'pos'], edge_attrs=['edge_attr'])
        Gp = nx.operators.compose_all([G.subgraph(g)
                                      for g in nx.components.connected_components(G) if len(g) >= min_nodes])
        output = from_networkx(Gp, group_node_attrs=['x', 'pos'], group_edge_attrs=['edge_attr'])
        output.pos = output.x[:, -2:]
        output.x = output.x[:, :-2]
        return output

    def __call__(self, data):
        return self.forward(data)


def far_mass(r, r_dist, p):
    return p if r_dist < r else 0
