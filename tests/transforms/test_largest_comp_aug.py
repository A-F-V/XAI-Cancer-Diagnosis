import numpy as np
from src.transforms.graph_augmentation.largest_component import LargestComponent
from torch_geometric.data import Data
import torch


def test_largest_comp_aug():
    lc = LargestComponent()
    x = torch.Tensor([[0, 0], [1, 1], [2, 2], [3, 3]])
    edge = torch.Tensor([[3, 1], [1, 2], [1, 3], [2, 1]]).t()
    pos = torch.Tensor([[0, 1], [1, 2], [2, 3], [3, 4]])
    graph = Data(x=x, edge_index=edge, pos=pos)

    v = lc(graph)

    assert v.x.shape[0] == 3
    assert v.x[0][0] == 1
