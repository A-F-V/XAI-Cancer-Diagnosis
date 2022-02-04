import torch
from torchvision.transforms import ToTensor
import pytorch_lightning as pl
from torch import optim, nn
from torch.nn import functional as F


class CellAutoEncoder(pl.LightningModule):
    def __init__(self, img_size=64, num_steps=0, train_loader=None, val_loader=None, **kwargs):
        super(CellAutoEncoder, self).__init__()
        self.args = kwargs
        self.img_size = img_size
        self.learning_rate = kwargs["START_LR"]

        self.num_steps = num_steps
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.encoder = nn.Sequential(
            Conv(3, 9),
            Conv(9, 9),
            Conv(9, 9),
            Conv(9, 9),
            nn.Flatten(),


            nn.Linear((img_size//16)**2*9, 72),
            nn.ReLU(True),
            nn.BatchNorm1d(72),
            nn.Linear(72, 21),
            nn.ReLU(True),
            nn.BatchNorm1d(21)
        )
        self.decoder = nn.Sequential(
            nn.Linear(21, 72),
            nn.ReLU(True),
            nn.BatchNorm1d(72),
            nn.Linear(72, (img_size//16)**2*9),
            nn.ReLU(True),
            nn.BatchNorm1d((img_size//16)**2*9),
            nn.Unflatten(0, (9, (img_size//16), (img_size//16))),
            DeConv(9, 9),
            DeConv(9, 9),
            DeConv(9, 9),
            DeConv(9, 3)

        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def encode(self, x):
        return self.encoder(x)

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
        cells, _ = train_batch
        cell_hat = self.forward(cells)

        loss = F.binary_cross_entropy(cells,cell_hat)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        cells, _ = val_batch
        cell_hat = self.forward(cells)

        loss = F.binary_cross_entropy(cells,cell_hat)
        self.log("val_loss", loss)
        return loss

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader


def Conv(in_c, out_c):
    return nn.Sequential(nn.Conv2d(in_c, out_channel=out_c, kernel_size=3, stride=1, padding=1),
                         nn.ReLU(True),
                         nn.BatchNorm2d(out_c),
                         nn.MaxPool2d((2, 2), stride=2))


def DeConv(in_c, out_c):
    return nn.Sequential(nn.Conv2d(in_c, out_channel=out_c, kernel_size=3, stride=1, padding=1),
                         nn.ReLU(True),
                         nn.BatchNorm2d(out_c),
                         nn.MaxUnpool2d((2, 2), stride=2))
