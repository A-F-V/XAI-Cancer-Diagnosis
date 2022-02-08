
import pytorch_lightning as pl
from torch_geometric.nn import TopKPooling, PNAConv, BatchNorm, global_mean_pool, global_max_pool, GCNConv
from torch.nn import ModuleList, Sequential, Linear, ReLU, BatchNorm1d, Softmax
from torch.nn.functional import nll_loss
import torch
from torch import optim, nn


class CellGraphSignatureGNN(nn.Module):
    def __init__(self, node_width, **config):
        super(CellGraphSignatureGNN, self).__init__()
        self.args = config
        self.encoder = ModuleList([GCNConv(in_channels=node_width if depth == 0 else config["WIDTH"], out_channels=[
                                  "WIDTH"], improved=True) for depth in range(config["LAYERS"])])
        self.global_pool = global_mean_pool if config["GLOBAL_POOL"] == "MAX" else global_mean_pool

    def forward(self, x, edge_index, edge_attr, batch):
        edge_attr = edge_attr.squeeze()
        for layer in self.encoder:
            x = layer.forward(x=x, edge_index=edge_index, edge_weight=edge_attr)
        return self.global_pool(x, batch)
