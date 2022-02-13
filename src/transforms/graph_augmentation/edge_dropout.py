import torch.nn as nn
import torch
from torch_geometric.transforms import BaseTransform


class EdgeDropout(BaseTransform):
    def __init__(self, p=0.001, p_mass=None):  # set p-mass not to use p
        super().__init__()
        self.p_mass = p_mass
        self.p = p

    def forward(self, graph):
        e_ind, e_w = graph.edge_index, graph.edge_attr
        if(self.p_mass != None):
            p_mass = self.p_mass(e_w[range(0, e_w.shape[0], 2), :])
        else:
            p_mass = torch.zeros(e_ind.shape[1]//2)+self.p
        keep = torch.bernoulli(1-p_mass).repeat_interleave(2).nonzero().squeeze()
        e_ind = e_ind.index_select(1, keep)
        graph.edge_index = e_ind
        if (e_w != None):
            e_w = e_w.index_select(0, keep)
            graph.edge_attr = e_w
            assert e_w.shape[0] == e_ind.shape[1]
        return graph

    def __call__(self, data):
        return self.forward(data)


def far_mass(r, r_dist, p):
    return p if r_dist < r else 0
