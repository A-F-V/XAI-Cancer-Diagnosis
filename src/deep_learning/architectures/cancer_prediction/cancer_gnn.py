
from torch import optim, Tensor, softmax
import torch
from torch.nn.functional import nll_loss, sigmoid, log_softmax, cross_entropy, one_hot, mse_loss
from torch.nn import ModuleList, Sequential, Linear, ReLU, BatchNorm1d, Softmax, Dropout, LeakyReLU, ModuleDict, Parameter, LayerNorm
from torch_geometric.nn import TopKPooling, PNAConv, BatchNorm, global_mean_pool, global_max_pool, GIN, GAT, GCNConv, TopKPooling, MessagePassing, GCN
import pytorch_lightning as pl
from torch.nn import Sequential as Seq, Linear as Lin
from src.deep_learning.architectures.components.gcnx import GCNx
from src.deep_learning.architectures.cancer_prediction.cell_encoder import CellEncoder
import os
from src.transforms.graph_construction.node_embedding import generate_node_embeddings


class CancerGNN(pl.LightningModule):
    def __init__(self, img_size=64, num_steps=0, train_loader=None, val_loader=None, pre_encoded=True, **config):
        super(CancerGNN, self).__init__()
        self.args = dict(config)
        self.img_size = img_size
        self.learning_rate = config["START_LR"] if "START_LR" in config else 1e-3
        if not pre_encoded:
            self.node_embedder_model = CellEncoder.load_from_checkpoint(os.path.join("model", "CellEncoder.ckpt"))
            self.node_embedder_model.eval()
            self.node_embedder_model.requires_grad_(False)
        self.num_steps = num_steps
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.height = self.args["HEIGHT"] if "HEIGHT" in self.args else 2
        self.width = self.args["WIDTH"] if "WIDTH" in self.args else 16
        self.gnn = GCNx(input_width=312, hidden_width=self.width, output_width=4, conv_depth=self.height)
        self.predictor = Seq(Dropout(p=0.4),
                             Lin(self.width*2, self.width),
                             BatchNorm1d(self.width, momentum=0.01),
                             ReLU(),
                             Dropout(p=0),
                             Lin(self.width, 4))
        # self.grader = Seq(Dropout(p=0.3),
        #                  Lin(self.width*2, self.width),
        #                  BatchNorm1d(self.width, momentum=0.01),
        #                  ReLU(),
        #                  Dropout(p=0.0),
        #                  Lin(self.width, 1),
        #                  ReLU())
        self.pre_encoded = pre_encoded

    def predict(self, graph):
        return self.forward(graph.x, graph.edge_index, torch.zeros(len(graph.x), dtype=torch.int64).to(graph.x.device))

    def forward(self, x, edge_index, batch):
        # TEMPORARY
        if x.shape[1] == 315:
            x = x[:, 3:]
       # x = self.inn(x)
        r = self.gnn(x, edge_index,  batch)
        return self.predictor(r)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate, eps=1e-5, weight_decay=1e-2)
        if self.args["ONE_CYCLE"]:
            lr_scheduler = optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=self.args['MAX_LR'], total_steps=self.num_steps,  three_phase=True)
            return [optimizer], [{'scheduler': lr_scheduler, 'interval': 'step'}]
        else:
            return optimizer

    def on_train_start(self):
        self.logger.log_hyperparams(self.args)

    def training_step(self, train_batch, batch_idx):
        x, edge_index, num_neighbours, cell_types, y, batch = train_batch.x, train_batch.edge_index, train_batch.num_neighbours, train_batch.categories, train_batch.y, train_batch.batch
        if not self.pre_encoded:
            glcm = train_batch.glcm
            x = generate_node_embeddings(imgs=x, resnet_encoder=self.node_embedder_model,
                                         num_neighbours=num_neighbours, cell_types=cell_types,
                                         glcm=glcm)

        y_hat = self.forward(x, edge_index, batch)
        y_hat_canc = to_cancer_scores(y_hat)
        y_canc = (y <= 1).to(dtype=torch.int64)

        four_loss = cross_entropy(y_hat, y)
        loss = four_loss  # + grade_loss  # + two_loss

        pred_cat = y_hat.argmax(dim=1)

        canc_pred = (pred_cat <= 1).float()
        canc_grd = (y <= 1).float()
        acc = (pred_cat == y).float().mean()
        canc_acc = (canc_pred == canc_grd).float().mean()

        self.log("four_loss", four_loss)
        self.log("train_loss", loss)
        self.log("train_acc", acc)
        self.log("train_canc_acc", canc_acc)

        # print(self.steepness.data)
        return {"loss": loss, "train_acc": acc, "train_canc_acc": canc_acc}

    def validation_step(self, val_batch, batch_idx):
        x, edge_index, num_neighbours, cell_types, y, batch = val_batch.x, val_batch.edge_index, val_batch.num_neighbours, val_batch.categories, val_batch.y, val_batch.batch

        if not self.pre_encoded:
            glcm = val_batch.glcm
            x = generate_node_embeddings(imgs=x, resnet_encoder=self.node_embedder_model,
                                         num_neighbours=num_neighbours, cell_types=cell_types,
                                         glcm=glcm)

        y_hat = self.forward(x, edge_index, batch)

        y_hat_canc = to_cancer_scores(y_hat)
        y_canc = (y <= 1).to(dtype=torch.int64)

        four_loss = cross_entropy(y_hat, y)

        loss = four_loss  # + grade_loss  # + two_loss

        pred_cat = y_hat.argmax(dim=1)

        canc_pred = (pred_cat <= 1).float()
        canc_grd = (y <= 1).float()
        acc = (pred_cat == y).float().mean()
        canc_acc = (canc_pred == canc_grd).float().mean()

        self.log("val_loss", loss)
        self.log("val_four_loss", four_loss)
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


def to_cancer_scores(y):
    assert len(y.shape) == 2
    assert y.shape[1] == 4
    output = torch.zeros(y.shape[0], 2)
    output[:, 0] = y[:, 0] + y[:, 1]
    output[:, 1] = y[:, 2] + y[:, 3]
    output = output.to(y.device)
    assert output.device == y.device
    return output
