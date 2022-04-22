import torch
from torchvision.transforms import ToTensor
import pytorch_lightning as pl
from torch import optim, nn, Tensor
from torch.nn import functional as F
from src.vizualizations.image_viz import plot_images
from src.utilities.img_utilities import tensor_to_numpy
from src.utilities.mlflow_utilities import log_plot
import torch
from src.utilities.pytorch_utilities import incremental_forward
from torchvision.transforms import RandomVerticalFlip, RandomHorizontalFlip, ColorJitter, Compose
from src.datasets.BACH_Cells import BACH_Cells
import os
from torch.utils.data import DataLoader, RandomSampler
from src.utilities.pytorch_utilities import random_subset
from src.utilities.tensor_utilties import one_hot_cartesian_product


# todo rename

class CellEncoder(pl.LightningModule):
    def __init__(self, train_loader, val_loader, img_size=64, num_steps=0, **kwargs):
        super(CellEncoder, self).__init__()
        self.args = kwargs
        self.img_size = img_size
        self.learning_rate = kwargs["START_LR"] if "START_LR" in kwargs else 1e-3
        self.num_steps = num_steps
        # self.data_set = BACH_Cells(data_set_path, img_size=self.img_size, **kwargs)
        # self.data_loader = DataLoader(
        #    self.data_set, batch_size=kwargs["BATCH_SIZE"], shuffle=True, num_workers=kwargs["NUM_WORKERS"])

        self.train_loader = train_loader
        self.val_loader = val_loader

        # self.encoder = UNET_Encoder(in_channels=3, out_channels=64, num_steps=self.num_steps, **kwargs)
        # self.decoder = UNET_Decoder(in_channels=64, out_channels=3, num_steps=self.num_steps, **kwargs)

        self.num_steps = num_steps
        self.width = kwargs["WIDTH"] if "WIDTH" in kwargs else 32
        self.dropout = kwargs["DROPOUT"] if "DROPOUT" in kwargs else 0

        self.encodercnn = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_resnet50', pretrained=True)
        self.encoderann = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(1000, (1000*self.width)//32),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm1d((1000*self.width)//32),
            nn.Dropout(self.dropout),
            nn.Linear((1000*self.width)//32, (200*self.width)//32),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm1d((200*self.width)//32),
            nn.Dropout(self.dropout),
            nn.Linear((200*self.width)//32, (40*self.width)//32),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm1d((40*self.width)//32))

        self.predictor = nn.Sequential(nn.Linear((40*self.width)//32, 4*5))  # NO SOFTMAX

    def forward(self, x):
        enc1 = self.encodercnn(x)
        enc2 = self.encoderann(enc1)
        pred = self.predictor(enc2)
        return enc2, pred

    @ incremental_forward(512)
    def forward_pred(self, x):
        return self.forward(x)[0]

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate,
                               eps=1e-5, weight_decay=self.args["WEIGHT_DECAY"])
        if self.args["ONE_CYCLE"]:
            lr_scheduler = optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=self.args['MAX_LR'], total_steps=self.num_steps,  three_phase=True)
            return [optimizer], [{'scheduler': lr_scheduler, 'interval': 'step'}]
        else:
            return optimizer

    def on_train_start(self):
        self.logger.log_hyperparams(self.args)

    def training_step(self, train_batch, batch_idx):
        cells, diag_hot, cell_type_hot = train_batch["img"], train_batch["diagnosis"].int(
        ), train_batch['cell_type'].int()

        y = one_hot_cartesian_product(diag_hot, cell_type_hot)
        y_cat = categorise(y)
        _, y_hat = self.forward(cells)

        loss = F.cross_entropy(y_hat, y_cat)

        overall_pred_label = y_hat.argmax(dim=1)
        cell_type_pred = overall_pred_label % 5
        diag_pred = overall_pred_label//5

        acc = (overall_pred_label == y_cat).float().mean()
        cell_type_acc = (cell_type_pred == categorise(cell_type_hot)).float().mean()
        diag_acc = (diag_pred == categorise(diag_hot)).float().mean()
        self.log("train_acc", acc)
        self.log("train_cell_type_acc", cell_type_acc)
        self.log("train_diag_acc", diag_acc)
        # pred_cat = y_hat.argmax(dim=1)
        # canc_pred = (torch.where(pred_cat.eq(0) | pred_cat.eq(3), 0, 1)).float()
        # canc_grd = (torch.where(y.eq(0) | y.eq(3), 0, 1)).float()
        # acc = (pred_cat == y).float().mean()
        # canc_acc = (canc_pred == canc_grd).float().mean()
        # self.log("train_acc", acc)
        # self.log("train_canc_acc", canc_acc)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        cells, diag_hot, cell_type_hot = val_batch["img"], val_batch["diagnosis"].int(
        ), val_batch['cell_type'].int()

        y = one_hot_cartesian_product(diag_hot, cell_type_hot)
        y_cat = categorise(y)
        _, y_hat = self.forward(cells)

        loss = F.cross_entropy(y_hat, y_cat)

        overall_pred_label = y_hat.argmax(dim=1)
        cell_type_pred = overall_pred_label % 5
        diag_pred = overall_pred_label//5

        acc = (overall_pred_label == y_cat).float().mean()
        cell_type_acc = (cell_type_pred == categorise(cell_type_hot)).float().mean()
        diag_acc = (diag_pred == categorise(diag_hot)).float().mean()
        self.log("val_acc", acc)
        self.log("val_cell_type_acc", cell_type_acc)
        self.log("val_diag_acc", diag_acc)
        # pred_cat = y_hat.argmax(dim=1)
        # canc_pred = (torch.where(pred_cat.eq(0) | pred_cat.eq(3), 0, 1)).float()
        # canc_grd = (torch.where(y.eq(0) | y.eq(3), 0, 1)).float()
        # acc = (pred_cat == y).float().mean()
        # canc_acc = (canc_pred == canc_grd).float().mean()
        # self.log("train_acc", acc)
        # self.log("train_canc_acc", canc_acc)
        self.log("val_loss", loss)
        return loss

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader


def categorise(t: Tensor):
    if len(t.shape) == 1:
        t = t.unsqueeze(0)
    return t.argmax(dim=1)
