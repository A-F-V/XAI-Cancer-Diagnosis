
import pytorch_lightning as pl
from torch_geometric.nn import TopKPooling, PNAConv, BatchNorm, global_mean_pool, global_max_pool, GIN
from torch.nn import ModuleList, Sequential, Linear, ReLU, BatchNorm1d, Softmax
from torch.nn.functional import nll_loss, sigmoid, log_softmax
import torch
from torch import optim, Tensor


class SimpleGNN(pl.LightningModule):
    def __init__(self, img_size=30, layers=8, num_batches=0, train_loader=None, val_loader=None, **kwargs):
        super(SimpleGNN, self).__init__()
        self.args = kwargs
        self.img_size = img_size
        self.learning_rate = kwargs["START_LR"]

        self.num_batches = num_batches
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.model = GIN(3*img_size**2, 3*img_size**2, num_layers=layers, jk='lstm', dropout=0.05, out_channels=300)
        self.predictor = Sequential(

            # BatchNorm1d(img_size**2*3),
            ReLU(),
            Linear(300, 100),
            # BatchNorm1d(100),
            ReLU(),
            Linear(100, 10),
            # BatchNorm1d(100),
            ReLU(),
            Linear(10, 4)

        )

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.model(x, edge_index)
        xp = global_max_pool(x, batch)
        return self.predictor(xp)

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
        output = self.forward(x, edge_index, edge_attr, batch)
        y_hat = log_softmax(output, dim=1)
        loss = nll_loss(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, edge_index, edge_attr, y, batch = val_batch.x, val_batch.edge_index, val_batch.edge_attr, val_batch.y, val_batch.batch
        output = self.forward(x, edge_index, edge_attr, batch)
        y_hat = log_softmax(output, dim=1)
        loss = nll_loss(y_hat, y)
        self.log("val_loss", loss)
        return loss

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader
