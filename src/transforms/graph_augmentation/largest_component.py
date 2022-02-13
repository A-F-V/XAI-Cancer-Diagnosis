
import torch.nn as nn
import torch
from torch_geometric.transforms import BaseTransform
import networkx as nx
from torch_geometric.utils import to_networkx
import torch_geometric
from collections import defaultdict


def from_networkx_fast(G, group_node_attrs=None,
                       group_edge_attrs=None):
    r"""From the Pytorch Geometric, but without the time consuming check
    """

    G = nx.convert_node_labels_to_integers(G)
    if not nx.is_directed(G):
        G = G.to_directed(as_view=True)  # NEED TO ADD TO PREVENT DEEP COPYING LOTS OF IMAGES

    edges = list(G.edges)

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    data = defaultdict(list)

    if G.number_of_nodes() > 0:
        node_attrs = list(next(iter(G.nodes(data=True)))[-1].keys())
    else:
        node_attrs = {}

    if G.number_of_edges() > 0:
        edge_attrs = list(next(iter(G.edges(data=True)))[-1].keys())
    else:
        edge_attrs = {}

    for i, (_, feat_dict) in enumerate(G.nodes(data=True)):
        if set(feat_dict.keys()) != set(node_attrs):
            raise ValueError('Not all nodes contain the same attributes')
        for key, value in feat_dict.items():
            data[str(key)].append(value)

    for i, (_, _, feat_dict) in enumerate(G.edges(data=True)):
        if set(feat_dict.keys()) != set(edge_attrs):
            raise ValueError('Not all edges contain the same attributes')
        for key, value in feat_dict.items():
            key = f'edge_{key}' if key in node_attrs else key
            data[str(key)].append(value)

    for key, value in data.items():
        try:
            data[key] = torch.tensor(value)
        except ValueError:
            pass

    data['edge_index'] = edge_index.view(2, -1)
    data = torch_geometric.data.Data.from_dict(data)

    if group_node_attrs is all:
        group_node_attrs = list(node_attrs)
    if group_node_attrs is not None:
        xs = []
        for key in group_node_attrs:
            x = data[key]
            x = x.view(-1, 1) if x.dim() <= 1 else x
            xs.append(x)
            del data[key]
        data.x = torch.cat(xs, dim=-1)

    if group_edge_attrs is all:
        group_edge_attrs = list(edge_attrs)
    if group_edge_attrs is not None:
        xs = []
        for key in group_edge_attrs:
            key = f'edge_{key}' if key in node_attrs else key
            x = data[key]
            x = x.view(-1, 1) if x.dim() <= 1 else x
            xs.append(x)
            del data[key]
        data.edge_attr = torch.cat(xs, dim=-1)

    if data.x is None and data.pos is None:
        data.num_nodes = G.number_of_nodes()

    return data


class LargestComponent(BaseTransform):
    def __init__(self):  # set p-mass not to use p
        super().__init__()

    def forward(self, graph):
        G = to_networkx(graph, to_undirected=True, node_attrs=['x', 'pos'], edge_attrs=['edge_attr'])
        Gp = G.subgraph(max(nx.connected_components(G), key=len)).copy()
        output = from_networkx_fast(Gp, group_node_attrs=['x', 'pos'], group_edge_attrs=['edge_attr'])
        output.pos = output.x[:, -2:]
        output.x = output.x[:, :-2]
        output.y = graph.y
        return output

    def __call__(self, data):
        return self.forward(data)
