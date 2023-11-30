
from torch import optim, Tensor, softmax
import torch
from torch.nn.functional import nll_loss, sigmoid, log_softmax, cross_entropy, one_hot, mse_loss
from torch.nn import ModuleList, Sequential, Linear, ReLU, BatchNorm1d, Softmax, Dropout, LeakyReLU, ModuleDict, Parameter, LayerNorm
from torch_geometric.nn import TopKPooling, PNAConv, BatchNorm, global_mean_pool, global_max_pool, GIN, GAT, GCNConv, TopKPooling, MessagePassing, GCN
import pytorch_lightning as pl
from torch.nn import Sequential as Seq, Linear as Lin
from src.deep_learning.architectures.components.gcnx_refactored import GCNRefx
from src.deep_learning.architectures.cancer_prediction.cell_encoder import CellEncoder
import os
from src.transforms.graph_construction.node_embedding import generate_node_embeddings
from src.deep_learning.architectures.components.cem import ConceptEncoderModule
from lens.models.mu_nn import XMuNN
from src.deep_learning.metrics.wasserstein_distance import WassersteinLoss
# TODO: Are we outputting the correct classes here?


class ExplainableCancerGNN(pl.LightningModule):
    def __init__(self, img_size=64, num_steps=0, train_loader=None, val_loader=None, **config):
        super(ExplainableCancerGNN, self).__init__()
        # Args
        self.args = dict(config)
        self.img_size = img_size
        self.learning_rate = config["START_LR"] if "START_LR" in config else 1e-3
        self.num_steps = num_steps
        self.height = self.args["HEIGHT"] if "HEIGHT" in self.args else 2
        self.width = self.args["WIDTH"] if "WIDTH" in self.args else 16
        self.concept_width = self.args["CONCEPT_WIDTH"] if "CONCEPT_WIDTH" in self.args else 16
        self.input_dropout = self.args["INPUT_DROPOUT"] if "INPUT_DROPOUT" in self.args else 0.1
        self.l1_weight = self.args["L1_WEIGHT"] if "L1_WEIGHT" in self.args else 0.01

        # Other
        self.train_loader = train_loader
        self.val_loader = val_loader
        # Arch
        self.gnn = GCNRefx(input_width=312, hidden_width=self.width, output_width=4,
                           conv_depth=self.height, input_dropout=self.input_dropout)
        self.cem = Seq(BatchNorm1d(self.width, momentum=0.01), ReLU(), Lin(
            self.width, self.concept_width), ConceptEncoderModule(self.concept_width))

        # Graph Mean pooling
        self.pool = global_mean_pool

        self.lens = XMuNN(4, self.concept_width, [
                          self.concept_width//2, self.concept_width//4], loss=WassersteinLoss(), l1_weight=self.l1_weight)
        # self.grader = Seq(Dropout(p=0.3),
        #                  Lin(self.width*2, self.width),
        #                  BatchNorm1d(self.width, momentum=0.01),
        #                  ReLU(),
        #                  Dropout(p=0.0),
        #                  Lin(self.width, 1),
        #                  ReLU())

    def predict(self, graph):
        return self.forward(graph.x, graph.edge_index, torch.zeros(len(graph.x), dtype=torch.int64).to(graph.x.device))

    def predict_proba(self, graph):
        return softmax(self.predict(graph)["pred"], dim=1)

    def forward(self, x, edge_index, batch):
        x = self.gnn(x, edge_index, batch)
        x = self.cem(x)
        r = self.pool(x, batch)
        y_hat = self.lens(r)
        result = {"node_concepts": x, "graph_concepts": r, "pred": y_hat}
        return result

    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.parameters(), lr=self.learning_rate, eps=1e-5, weight_decay=1e-2)
        if self.args["ONE_CYCLE"]:
            lr_scheduler = optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=self.args['MAX_LR'], total_steps=self.num_steps,  three_phase=True)
            return [optimizer], [{'scheduler': lr_scheduler, 'interval': 'step'}]
        else:
            return optimizer

    def on_train_start(self):
        self.logger.log_hyperparams(self.args)

    def forward_step(self, batch, batch_idx, step_type="train"):
        x, edge_index, num_neighbours, cell_types, y, batch = batch.x, batch.edge_index, batch.num_neighbours, batch.categories, batch.y, batch.batch

        result = self.forward(x, edge_index, batch)
        y_hat = result["pred"]
        node_concepts = result["node_concepts"]
        graph_concepts = result["graph_concepts"]

        loss = self.lens.get_loss(y_hat, y)
        ce_loss = cross_entropy(y_hat, y)
        l1_loss = loss - ce_loss

        pred_cat = y_hat.argmax(dim=1)

        acc = (pred_cat == y).float().mean()

        self.log(f"{step_type}_loss", loss)
        self.log(f"{step_type}_acc", acc)
        self.log(f"{step_type}_ce_loss", ce_loss)
        self.log(f"{step_type}_l1_loss", l1_loss)

        # print(self.steepness.data)
        return {"loss": loss, "acc": acc}

    def training_step(self, batch, batch_idx):
        return self.forward_step(batch, batch_idx, step_type="train")

    def validation_step(self, batch, batch_idx):
        return self.forward_step(batch, batch_idx, step_type="val")

    def epoch_end_step(self, outputs, step_type="train"):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        avg_acc = torch.stack([x["acc"] for x in outputs]).mean()
        self.log(f"ep/{step_type}_loss", avg_loss)
        self.log(f"ep/{step_type}_acc", avg_acc)

    def train_epoch_end(self, outputs):
        self.epoch_end_step(outputs, step_type="train")

    def validation_epoch_end(self, outputs):
        self.epoch_end_step(outputs, step_type="val")

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader


def create_linear_predictor(**config):
    widths = [config["WIDTH"], max(
        4, config["WIDTH"]//2), max(4, config["WIDTH"]//4)]
    layers = []
    for i in range(config["FFN_DEPTH"]):
        # layers.append(Dropout(config["DROPOUT"], inplace=True))
        layers.append(BatchNorm(widths[i]))
        layers.append(ReLU(inplace=True))
        layers.append(Linear(widths[i], 4 if i+1 ==
                      config["FFN_DEPTH"] else widths[i+1]))
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
