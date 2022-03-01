
import pytorch_lightning as pl
from torch_geometric.nn import TopKPooling, PNAConv, BatchNorm, global_mean_pool, global_max_pool, GIN, GAT, GCN, GCNConv
from torch.nn import ModuleList, Sequential, Linear, ReLU, BatchNorm1d, Softmax, Dropout, LeakyReLU, ModuleDict, Parameter
from torch.nn.functional import nll_loss, sigmoid, log_softmax, cross_entropy, one_hot
import torch
from torch import optim, Tensor, softmax
from src.transforms.graph_construction.graph_extractor import mean_pixel_extraction, principle_pixels_extraction
from src.model.architectures.cancer_prediction.cgs_gnn import CellGraphSignatureGNN
from src.model.architectures.cancer_prediction.cell_autoencoder import Conv
import torch.nn as nn
from src.utilities.pytorch_utilities import incremental_forward
from src.model.metrics.graph_agreement import hard_agreement


class PredGNN(pl.LightningModule):
    def __init__(self, img_size=64, num_steps=0, train_loader=None, val_loader=None, **config):
        super(PredGNN, self).__init__()
        self.args = dict(config)
        self.img_size = img_size
        self.learning_rate = config["START_LR"]
        print(self.learning_rate)

        self.num_steps = num_steps
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.layers = self.args["LAYERS"]

        self.model = ModuleList([ModuleDict({"pre_trans": Linear(4, 4), "pre_act": LeakyReLU(),
                                "conv": GCNConv(4, 4), "post_act": Softmax(dim=1)}) for i in range(self.layers)])
        self.pool = global_max_pool if self.args["GLOBAL_POOL"] == "MAX" else global_mean_pool

    def forward(self, x, edge_index, edge_attr, batch):
        edge_attr = (50**2)/(edge_attr.squeeze()**(2))
        # , edge_weight=edge_attr)
        # x = one_hot(x.argmax(dim=1), num_classes=4).float()
        for i in range(self.layers):
           # x = self.model[i]["pre_act"](self.model[i]["pre_trans"](x))
            x = self.model[i]["conv"](x=x, edge_index=edge_index)
           # x = self.model[i]["post_act"](x*1)
        x_pool = self.pool(x, batch)
        soft = softmax(x_pool, dim=1)
        return soft

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate, eps=1e-5, weight_decay=0.0)
        if self.args["ONE_CYCLE"]:
            lr_scheduler = optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=self.args['MAX_LR'], total_steps=self.num_steps,  three_phase=True)
            return [optimizer], [{'scheduler': lr_scheduler, 'interval': 'step'}]
        else:
            return optimizer

    def on_train_start(self):
        self.logger.log_hyperparams(self.args)

    def training_step(self, train_batch, batch_idx):
        x, edge_index, edge_attr, y, batch = train_batch.x, train_batch.edge_index, train_batch.edge_attr, train_batch.y, train_batch.batch
        output = self.forward(x, edge_index, edge_attr, batch)
        loss = nll_loss(torch.log(output), y)
        self.log("train_loss", loss)
        # print(self.steepness.data)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, edge_index, edge_attr, y, batch = val_batch.x, val_batch.edge_index, val_batch.edge_attr, val_batch.y, val_batch.batch
        output = self.forward(x, edge_index, edge_attr, batch)
        loss = nll_loss(torch.log(output), y)
        pred_cat = output.argmax(dim=1)

        canc_pred = (torch.where(pred_cat.eq(0) | pred_cat.eq(3), 0, 1)).float()
        canc_grd = (torch.where(y.eq(0) | y.eq(3), 0, 1)).float()
        acc = (pred_cat == y).float().mean()
        canc_acc = (canc_pred == canc_grd).float().mean()

        self.log("val_loss", loss)
        self.log("val_acc", acc)
        self.log("val_canc_acc", canc_acc)
        return {'val_loss': loss, 'val_acc': acc, 'val_canc_acc': canc_acc}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        avg_acc = torch.stack([x["val_acc"] for x in outputs]).mean()
        avg_canc_acc = torch.stack([x["val_canc_acc"] for x in outputs]).mean()
        self.log("ep/val_loss", avg_loss)
        self.log("ep/val_acc", avg_acc)
        self.log("ep/val_canc_acc", avg_canc_acc)

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader


def create_linear_predictor(**config):
    widths = [config["WIDTH"], max(4, config["WIDTH"]//2), max(4, config["WIDTH"]//4)]
    layers = []
    for i in range(config["FFN_DEPTH"]):
        # layers.append(Dropout(config["DROPOUT"], inplace=True))
        layers.append(BatchNorm(widths[i]))
        layers.append(ReLU(inplace=True))
        layers.append(Linear(widths[i], 4 if i+1 == config["FFN_DEPTH"] else widths[i+1]))
    layers.append(Softmax(dim=1))
    return Sequential(*layers)


class CellEncoder(nn.Module):  # todo make even more customizable
    def __init__(self, batch_size=16, **config):
        super(CellEncoder, self).__init__()
        self.bottleneck = config["ENCODER_BOTTLENECK"]
        depth = config["ENCODING_DEPTH"]
        self.conv_encoder = nn.Sequential(
            nn.BatchNorm2d(3),
            Conv(3, 3, depth=depth),
            Conv(3, 3, depth=depth),
            Conv(3, 3, depth=depth),
            Conv(3, 3, depth=depth),
            nn.Flatten(),


            nn.Linear(16*3, 20),
            nn.LeakyReLU(True),
            nn.BatchNorm1d(20),
            nn.Linear(20, self.bottleneck),
            nn.LeakyReLU(True),
            nn.BatchNorm1d(self.bottleneck)
        )
        self.batch_size = batch_size

    def forward(self, x):
        x_unflat = x.unflatten(1, (3, 64, 64))
        return self.conv_encoder(x_unflat)


class LinearEncoder(nn.Module):
    def __init__(self, **config):
        super(LinearEncoder, self).__init__()
        self.enc = nn.Linear(3*8*8, config["ENCODER_BOTTLENECK"])

    def forward(self, x):
        x = x.unflatten(1, (3, 64, 64))
        x = nn.AdaptiveAvgPool2d((3, 8, 8))(x).flatten(1)
        return self.enc(x)