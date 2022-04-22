from src.model.evaluation.graph_agreement import hard_agreement
from src.utilities.pytorch_utilities import incremental_forward
import torch.nn as nn
from src.model.architectures.cancer_prediction.cell_autoencoder import Conv
from src.model.architectures.cancer_prediction.cgs_gnn import CellGraphSignatureGNN
from src.transforms.graph_construction.graph_extractor import mean_pixel_extraction, principle_pixels_extraction
from torch import optim, Tensor, softmax
import torch
from torch.nn.functional import nll_loss, sigmoid, log_softmax, cross_entropy, one_hot
from torch.nn import ModuleList, Sequential, Linear, ReLU, BatchNorm1d, Softmax, Dropout, LeakyReLU, ModuleDict, Parameter
from torch_geometric.nn import TopKPooling, PNAConv, BatchNorm, global_mean_pool, global_max_pool, GIN, GAT, GCN, GCNConv, TopKPooling, MessagePassing
import pytorch_lightning as pl
from torch_geometric.nn import GINConv, Sequential as Seq, Linear as Lin
from src.model.architectures.components.gintopk import GINTopK


class CancerGNN(pl.LightningModule):
    def __init__(self, img_size=64, num_steps=0, train_loader=None, val_loader=None, **config):
        super(CancerGNN, self).__init__()
        self.args = dict(config)
        self.img_size = img_size
        self.learning_rate = config["START_LR"] if "START_LR" in config else 1e-3

        self.num_steps = num_steps
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.height = self.args["HEIGHT"]
        self.width = self.args["WIDTH"]
        self.model = GINTopK(input_width=40, hidden_width=self.width, output_width=4, conv_depth=self.height)

    def forward(self, x, edge_index, edge_attr, batch):
        return self.model(x, edge_index, edge_attr, batch)

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
        y_hat = self.forward(x, edge_index, edge_attr, batch)

        loss = cross_entropy(y_hat, y)

        pred_cat = y_hat.argmax(dim=1)

        canc_pred = (torch.where(pred_cat.eq(0) | pred_cat.eq(3), 0, 1)).float()
        canc_grd = (torch.where(y.eq(0) | y.eq(3), 0, 1)).float()
        acc = (pred_cat == y).float().mean()
        canc_acc = (canc_pred == canc_grd).float().mean()

        self.log("train_loss", loss)
        self.log("train_acc", acc)
        self.log("train_canc_acc", canc_acc)

        # print(self.steepness.data)
        return {"loss": loss, "train_acc": acc, "train_canc_acc": canc_acc}

    def validation_step(self, val_batch, batch_idx):
        x, edge_index, edge_attr, y, batch = val_batch.x, val_batch.edge_index, val_batch.edge_attr, val_batch.y, val_batch.batch
        y_hat = self.forward(x, edge_index, edge_attr, batch)

        loss = cross_entropy(y_hat, y)

        pred_cat = y_hat.argmax(dim=1)

        canc_pred = (torch.where(pred_cat.eq(0) | pred_cat.eq(3), 0, 1)).float()
        canc_grd = (torch.where(y.eq(0) | y.eq(3), 0, 1)).float()
        acc = (pred_cat == y).float().mean()
        canc_acc = (canc_pred == canc_grd).float().mean()

        self.log("val_loss", loss)
        self.log("val_acc", acc)
        return {'val_loss': loss, 'val_acc': acc, 'val_canc_acc': canc_acc}

    def train_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        avg_acc = torch.stack([x["train_acc"] for x in outputs]).mean()
        avg_canc_acc = torch.stack([x["train_canc_acc"] for x in outputs]).mean()
        self.log("ep/train_loss", avg_loss)
        self.log("ep/train_acc", avg_acc)
        self.log("ep/train_canc_acc", avg_canc_acc)

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
