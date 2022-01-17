
import pytorch_lightning as pl
from torch_geometric.nn import TopKPooling, PNAConv, BatchNorm, global_mean_pool, global_max_pool
from torch.nn import ModuleList, Sequential, Linear, ReLU, BatchNorm1d, Softmax
from torch.nn.functional import nll_loss
import torch
from torch import optim


class CancerNet(pl.LightningModule):
    def __init__(self, degree_dist, img_size=30, down_samples=1, tissue_radius=5, num_batches=0, train_loader=None, val_loader=None, **kwargs):
        super(CancerNet, self).__init__()
        self.args = kwargs
        self.img_size = img_size
        self.learning_rate = kwargs["START_LR"]

        self.num_batches = num_batches
        self.train_loader = train_loader
        self.val_loader = val_loader

        aggregators = ['mean', 'min', 'max', 'std']
        scalers = ['identity', 'amplification', 'attenuation']

        self.encoder = ModuleList()
        for i in range(down_samples+1):
            if i == down_samples:
                self.encoder.append(EncoderUnit(img_size**2*3, aggregators, scalers, degree_dist, tissue_radius, False))
            else:
                self.encoder.append(EncoderUnit(img_size**2*3, aggregators, scalers, degree_dist, tissue_radius, True))

        self.predictor = Sequential(

            BatchNorm1d(img_size**2*3),
            ReLU(),
            Linear(img_size**2*3, 100),
            BatchNorm1d(100),
            ReLU(),
            Linear(100, 10),
            BatchNorm1d(100),
            ReLU(),
            Linear(10, 4),
            Softmax()

        )

    def forward(self, x, edge_index, edge_attr, batch):
        for i in range(len(self.encoder)):
            x, edge_index, edge_attr, batch = self.encoder[i](x, edge_index, edge_attr, batch)
        x = global_max_pool(x, batch)
        return self.predictor(x)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate, eps=1e-5, weight_decay=0)
        if self.args["ONE_CYCLE"]:
            lr_scheduler = optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=self.args['MAX_LR'], total_steps=self.num_batches,  three_phase=True)
            return [optimizer], [{'scheduler': lr_scheduler, 'interval': 'step'}]
        else:
            return optimizer

    def on_train_start(self):
        self.logger.log_hyperparams(self.args)

    def training_step(self, train_batch, batch_idx):
        x, edge_index, edge_attr, y, batch = train_batch.x, train_batch.edge_index, train_batch.edge_attr, train_batch.y, train_batch.batch
        y_hat = self.forward(x, edge_index, edge_attr, batch)
        loss = nll_loss(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, edge_index, edge_attr, y, batch = val_batch.x, val_batch.edge_index, val_batch.edge_attr, val_batch.y, val_batch.batch
        y_hat = self.forward(x, edge_index, edge_attr, batch)
        loss = nll_loss(y_hat, y)
        self.log("val_loss", loss)
        return loss

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader


class EncoderUnit(pl.LightningModule):
    def __init__(self, channels, aggregators, scalers, deg, layers, donwsample=True, **args):
        super(EncoderUnit, self).__init__()
        self.pooling = TopKPooling(channels, ratio=0.5)
        self.convs = ModuleList()
        self.bn = ModuleList()
        for _ in range(layers):
            self.convs.append(PNAConv(channels, channels, aggregators=aggregators,
                                      scalers=scalers, deg=deg, edge_dim=1, pre_layers=0, post_layers=0))
            self.bn.append(BatchNorm(channels))
        self.downsample = donwsample

    def forward(self, x, edge_index, edge_attr, batch):
        x_prime = x
        for i in range(len(self.convs)):
            x_prime = self.convs[i](x_prime, edge_index, edge_attr)
            x_prime = self.bn[i](x_prime)
        if self.downsample:
            x_prime, edge_index, edge_attr, batch, _, _ = self.pooling(x_prime, edge_index, edge_attr, batch)
        return x_prime, edge_index, edge_attr, batch
