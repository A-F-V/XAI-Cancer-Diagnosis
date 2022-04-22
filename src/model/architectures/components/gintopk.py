from src.model.evaluation.graph_agreement import hard_agreement
from src.utilities.pytorch_utilities import incremental_forward
import torch.nn as nn
from src.model.architectures.cancer_prediction.cell_autoencoder import Conv
from src.model.architectures.cancer_prediction.cgs_gnn import CellGraphSignatureGNN
from src.transforms.graph_construction.graph_extractor import mean_pixel_extraction, principle_pixels_extraction
from torch import optim, Tensor, softmax
import torch
from torch.nn.functional import nll_loss, sigmoid, log_softmax, cross_entropy, one_hot
from torch.nn import ModuleList,  Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d, Softmax, Dropout, LeakyReLU, ModuleDict, Parameter
from torch_geometric.nn import TopKPooling, PNAConv, BatchNorm, global_mean_pool as gap, global_max_pool as gmp, GIN, GAT, GCN, GCNConv, TopKPooling, MessagePassing
import pytorch_lightning as pl
from torch_geometric.nn import GINConv


def _create_convolution(in_channels, out_channels):
    return GINConv(nn=Seq(Lin(in_channels, in_channels), ReLU(), Lin(in_channels, out_channels), ReLU()))


class GINTopK(torch.nn.Module):
    def __init__(self, input_width, hidden_width, output_width, conv_depth=4):
        super(GINTopK, self).__init__()
        self.iw = input_width
        self.hw = hidden_width
        self.ow = output_width

        self.conv_depth = conv_depth
        self.conv = ModuleList([_create_convolution(self.iw if i == 0 else self.hw, self.hw)
                               for i in range(self.conv_depth)])
        self.pool = ModuleList([TopKPooling(self.hw, ratio=0.8)
                               for i in range(self.conv_depth)])

        self.predictor = Seq(Lin(2*hidden_width, hidden_width),
                             ReLU(),
                             Dropout(),
                             Lin(hidden_width, hidden_width//2),
                             ReLU(),
                             Lin(hidden_width//2, output_width))

    def forward(self, x, edge_index, batch):

        readouts = []
        for i in range(self.conv_depth):
            x = self.conv[i](x, edge_index)
            x, edge_index, _, batch, _, _ = self.pool[i](x, edge_index, None, batch)
            r = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
            readouts.append(r)
        r = torch.sum(torch.stack(readouts, dim=1), dim=1)

        x = self.predictor(r)
        return x
