
import pytorch_lightning as pl
from torch_geometric.nn import TopKPooling, PNAConv, BatchNorm, global_mean_pool, global_max_pool
from torch_geometric.nn.conv import GCNConv
from torch.nn import ModuleList, Sequential, Linear, ReLU, Softmax, Dropout, ModuleDict, LeakyReLU
from torch.nn.functional import nll_loss
import torch
from torch import optim, nn


class CellGraphSignatureGNN(nn.Module):
    def __init__(self, **config):
        super(CellGraphSignatureGNN, self).__init__()
        self.args = config
        self.encoder = ModuleList(map(lambda lay: CGSLayer(config, lay), range(config["LAYERS"])))
        self.global_pool = global_max_pool if config["GLOBAL_POOL"] == "MAX" else global_mean_pool

    def forward(self, x, edge_index, edge_attr, batch):
        edge_attr = 1-edge_attr.squeeze()
        for level in self.encoder:
            x = level(x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch)
        return self.global_pool(x, batch)


class CGSLayer(nn.Module):

    def __init__(self, config, layer):
        super(CGSLayer, self).__init__()
        in_width = config["INPUT_WIDTH"] if layer == 0 else config["WIDTH"]
        self.model = ModuleDict()
        self.model["bn"] = BatchNorm(in_channels=in_width)
        if layer != 0:
            self.model["dp"] = Dropout(p=config["DROPOUT"])
        self.model["conv"] = GCNConv(in_channels=in_width, out_channels=config["WIDTH"], improved=True)
        self.model["act"] = LeakyReLU()

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.model["bn"](x)
        if "dp" in self.model:
            x = self.model["dp"](x)
        x = self.model["conv"](x=x, edge_index=edge_index, edge_weight=edge_attr)
        x = self.model["act"](x)
        return x
