
import pytorch_lightning as pl
from torch_geometric.nn import TopKPooling, PNAConv, BatchNorm, global_mean_pool, global_max_pool, GIN, GAT, GCN
from torch.nn import ModuleList, Sequential, Linear, ReLU, BatchNorm1d, Softmax, Dropout
from torch.nn.functional import nll_loss, sigmoid, log_softmax
import torch
from torch import optim, Tensor
from src.transforms.graph_construction.graph_extractor import mean_pixel_extraction, principle_pixels_extraction


def mean_pixel(X: Tensor):  # batched
    img_size = int((X.shape[1]//3)**0.5)
    pixel = X.unflatten(1, (3, img_size, img_size)).mean(dim=(2, 3))
    return pixel


class CancerPredictorGNN(pl.LightningModule):

    def reducer(self, name):
        if name == "MEAN_PIXEL":
            return mean_pixel, 3

    def __init__(self, img_size=30, num_steps=0, train_loader=None, val_loader=None, **config):
        super(CancerPredictorGNN, self).__init__()
        self.args = config
        self.img_size = img_size
        self.learning_rate = config["START_LR"]

        self.num_steps = num_steps
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.node_reducer, node_dim = self.reducer(config["NODE_REDUCER"])

        # self.model = GIN(3*img_size**2, 3*img_size**2//4, num_layers=layers, dropout=0.8, out_channels=300)'
        self.model = GCN(node_dim, 5, num_layers=config["LAYERS"], dropout=config["DROPOUT"], out_channels=5) if config["ARCH"] == "GCN" else GIN(
            node_dim, 5, num_layers=config["LAYERS"], dropout=config["DROPOUT"], out_channels=5)
        self.predictor = Sequential(

            # BatchNorm(300),
            ReLU(),
            Linear(5, 4),


        )

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.node_reducer(x)
        x = self.model(x, edge_index)  # , edge_weight=edge_attr)
        xp = global_mean_pool(x, batch)
        return self.predictor(xp)

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
        y_hat = log_softmax(output, dim=1)
        loss = nll_loss(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, edge_index, edge_attr, y, batch = val_batch.x, val_batch.edge_index, val_batch.edge_attr, val_batch.y, val_batch.batch
        output = self.forward(x, edge_index, edge_attr, batch)
        y_hat = log_softmax(output, dim=1)
        loss = nll_loss(y_hat, y)
        pred_cat = y_hat.argmax(dim=1)

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
