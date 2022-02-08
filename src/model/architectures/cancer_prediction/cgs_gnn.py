
import pytorch_lightning as pl
from torch_geometric.nn import TopKPooling, PNAConv, BatchNorm, global_mean_pool, global_max_pool
from torch_geometric.nn.conv import GCNConv
from torch.nn import ModuleList, Sequential, Linear, ReLU, BatchNorm1d, Softmax, Dropout
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
        edge_attr = edge_attr.squeeze()
        for level in self.encoder:
            x = level(x=x, edge_index=edge_index, edge_attr=edge_attr,batch=batch)
        return self.global_pool(x, batch)


class CGSLayer(nn.Module):

    def __init__(self, config, layer):
        super(CGSLayer, self).__init__()
        self.model = ModuleList()
        if layer != 0:
            self.model.append(Dropout(p=config["DROPOUT"]))
        self.model.append(GCNConv(in_channels=config["INPUT_WIDTH"] if layer ==
                                  0 else config["WIDTH"], out_channels=config["WIDTH"], improved=True))

    def forward(self, x, edge_index, edge_attr, batch):
        if len(self.model) == 2:
            x = self.model[0](x)
        return self.model[-1](x=x, edge_index=edge_index, edge_weight=edge_attr)
