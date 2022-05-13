
import pytorch_lightning as pl
from torch_geometric.nn import TopKPooling, PNAConv, BatchNorm, global_mean_pool, global_max_pool
from torch_geometric.nn.conv import GCNConv, GATConv
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

        for level in self.encoder:
            x = level(x=x, edge_index=edge_index, batch=batch, edge_attr=edge_attr)
        return self.global_pool(x, batch)


def create_layer(l_type, in_c, out_c, config):
    if l_type == "GAT":
        return GATConv(in_c, out_c,   edge_dim=1)
    elif l_type == "GCN":
        return GCNConv(in_channels=in_c, out_channels=out_c, improved=True)


class CGSLayer(nn.Module):

    def __init__(self, config, layer):
        super(CGSLayer, self).__init__()
        in_width = config["INPUT_WIDTH"] if layer == 0 else config["WIDTH"]
        self.config = config
        self.model = ModuleDict()
        self.model["bn"] = BatchNorm(in_channels=config["WIDTH"])
        if layer != 0:
            self.model["dp"] = Dropout(p=config["DROPOUT"])
        self.model["conv"] = create_layer(config["ARCH"], in_width, config["WIDTH"], config)
        self.model["act"] = LeakyReLU(inplace=True)

    def forward(self, x, edge_index, edge_attr, batch):
        if "dp" in self.model:
            x = self.model["dp"](x)

        if self.config["ARCH"] == "GAT":
            x = self.model["conv"](x=x, edge_index=edge_index, edge_attr=edge_attr)  # 1 for conv 1 for gat
        elif self.config["ARCH"] == "GCN":
            x = self.model["conv"](x=x, edge_index=edge_index, edge_weight=edge_attr)
        x = self.model["bn"](x)
        x = self.model["act"](x)
        return x
