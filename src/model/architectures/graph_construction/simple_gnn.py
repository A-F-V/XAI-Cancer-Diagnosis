from re import L
import pytorch_lightning as pl
from torch_geometric.nn import TopKPooling, PNAConv, BatchNorm, global_mean_pool, global_max_pool
from torch.nn import ModuleList, Sequential, Linear, ReLU, BatchNorm1d, Softmax
import torch


class CancerNet(pl.LightningModule):
    def __init__(self, degree_dist, img_size=30, down_samples=1, tissue_radius=5, **args):
        super(CancerNet, self).__init__()
        self.args = args
        self.model = None
        self.img_size = img_size

        aggregators = ['mean', 'min', 'max', 'std']
        scalers = ['identity', 'amplification', 'attenuation']

        self.encoder = ModuleList()
        for i in range(down_samples+1):
            if i == down_samples:
                self.encoder.append(EncoderUnit(img_size**2, aggregators, scalers, degree_dist, tissue_radius, False))
            else:
                self.encoder.append(EncoderUnit(img_size**2, aggregators, scalers, degree_dist, tissue_radius, True))

        self.predictor = Sequential(
            [
                BatchNorm1d(img_size**2),
                ReLU(),
                Linear(img_size**2, 100),
                BatchNorm1d(100),
                ReLU(),
                Linear(100, 10),
                BatchNorm1d(100),
                ReLU(),
                Linear(10, 4),
                Softmax()
            ]
        )

    def forward(self, x, edge_index, edge_attr, batch):
        for i in range(len(self.encoder)):
            x, edge_index, edge_attr, batch = self.encoder[i](x, edge_index, edge_attr, batch)
        x = global_max_pool(x, batch)
        return self.predictor(x)


class EncoderUnit(pl.LightningModule):
    def __init__(self, channels, aggregators, scalers, deg, layers, donwsample=True, **args):
        super(EncoderUnit, self).__init__()
        self.pooling = TopKPooling(channels, ratio=0.5)
        self.convs = ModuleList()
        self.bn = ModuleList()
        for _ in range(layers):
            self.convs.append(PNAConv(channels, channels, aggregators=aggregators,
                                      scalers=scalers, deg=deg, edge_dim=1))
            self.bn.append(BatchNorm(channels))
        self.downsample = donwsample

    def forward(self, x, edge_index, edge_attr, batch):
        x_prime = x
        for i in range(len(self.convs)):
            x_prime = self.convs[i](x_prime, edge_index, edge_attr)
            x_prime = self.bn[i](x_prime)
        if self.downsample:
            x_prime, edge_index, edge_attr, batch = self.pooling(x_prime, edge_index, edge_attr, batch)
        return x_prime, edge_index, edge_attr, batch
