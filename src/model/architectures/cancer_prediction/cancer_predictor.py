
import pytorch_lightning as pl
from torch_geometric.nn import TopKPooling, PNAConv, BatchNorm, global_mean_pool, global_max_pool, GIN, GAT, GCN
from torch.nn import ModuleList, Sequential, Linear, ReLU, BatchNorm1d, Softmax, Dropout, LeakyReLU
from torch.nn.functional import nll_loss, sigmoid, log_softmax, cross_entropy
import torch
from torch import optim, Tensor, softmax
from src.transforms.graph_construction.graph_extractor import mean_pixel_extraction, principle_pixels_extraction
from src.model.architectures.cancer_prediction.cgs_gnn import CellGraphSignatureGNN
from src.model.architectures.cancer_prediction.cell_autoencoder import Conv
import torch.nn as nn
from src.utilities.pytorch_utilities import incremental_forward


def mean_pixel(X: Tensor):  # batched
    pixels = X.unflatten(1, (3, -1)).mean(dim=2)
    return pixels


def constant(X: Tensor):
    return X.new_ones((X.shape[0], 1))


class CancerPredictorGNN(pl.LightningModule):

    def reducer(self, name):
        if name == "MEAN_PIXEL":
            return mean_pixel, 3
        if name == "CONSTANT":
            return constant, 1
        if name == "ENCODER":
            return self.node_encoder, self.args["ENCODER_BOTTLENECK"]

    def __init__(self, img_size=64, num_steps=0, train_loader=None, val_loader=None, **config):
        super(CancerPredictorGNN, self).__init__()
        self.args = dict(config)
        self.img_size = img_size
        self.learning_rate = config["START_LR"]

        self.num_steps = num_steps
        self.train_loader = train_loader
        self.val_loader = val_loader

        if self.args["NODE_REDUCER"] == "ENCODER":
            self.node_encoder = CellEncoder(**self.args)
            #self.node_encoder = LinearEncoder(**self.args)

        self.node_reducer, node_dim = self.reducer(config["NODE_REDUCER"])
        self.norm = BatchNorm(node_dim)

        # self.model = GIN(3*img_size**2, 3*img_size**2//4, num_layers=layers, dropout=0.8, out_channels=300)'
        self.args.update({"INPUT_WIDTH": node_dim})

        self.model = CellGraphSignatureGNN(**self.args)
        self.predictor = create_linear_predictor(**self.args)

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.node_reducer(x)
        x = self.norm(x)
        edge_attr = (50**2)/(edge_attr.squeeze()**(2))
        x_pooled = self.model(x, edge_index, edge_attr, batch)  # , edge_weight=edge_attr)
        return self.predictor(x_pooled)

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
        #layers.append(Dropout(config["DROPOUT"], inplace=True))
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
