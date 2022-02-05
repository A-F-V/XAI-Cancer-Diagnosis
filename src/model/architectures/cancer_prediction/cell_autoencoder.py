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
            nn.Unflatten(1, (9, (img_size//16), (img_size//16))),
            DeConv(9, 9, (img_size//8)),
            DeConv(9, 9, (img_size//4)),
            DeConv(9, 9, (img_size//2)),
            DeConv(9, 3, (img_size//1)),

        )
        self.predictor = nn.Sequential(

            nn.Linear(21, 14),
            nn.ReLU(),
            nn.BatchNorm1d(14),
            nn.Linear(14, 7),
            nn.ReLU(),
            nn.BatchNorm1d(7),
            nn.Linear(7, 4),
            nn.Softmax(dim=1))

    def forward(self, x):
        x = self.encoder(x)
        y = self.predictor(x)
        x = self.decoder(x)
        return x, y

    def encode(self, x):
        return self.encoder(x)

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
        cells, y = train_batch["img"], train_batch["diagnosis"].float()
        cell_hat, y_hat = self.forward(cells)

        loss = F.binary_cross_entropy(cells, cell_hat) + F.cross_entropy(y, y_hat)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        cells, y = val_batch['img'], val_batch["diagnosis"].float()
        cell_hat, y_hat = self.forward(cells)

        loss = F.binary_cross_entropy(cells, cell_hat) + F.cross_entropy(y, y_hat)

        self.log("val_loss", loss)
        return loss

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def on_epoch_end(self):
        if self.current_epoch != 0:
            if self.args["EPOCHS"] < 20 or self.current_epoch % ((self.args["EPOCHS"]+20)//10) == 0:

                sample = self.val_dataloader().dataset[0]
                # self.logger.experiment.log_artifact(local_path=gif_diag_path, artifact_path=f"Cell_Seg_{self.current_epoch}", run_id=self.logger.run_id)  # , "sliding_window_gif")


def Conv(in_c, out_c):
    return nn.Sequential(nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=3, stride=1, padding=1),
                         nn.ReLU(True),
                         nn.BatchNorm2d(out_c),
                         nn.Conv2d(in_channels=out_c, out_channels=out_c, kernel_size=3, stride=2, padding=1),
                         nn.ReLU(True),
                         nn.BatchNorm2d(out_c))


class DeConv(nn.Module):
    def __init__(self, in_c, out_c, out_img_dim=64):
        super(DeConv, self).__init__()
        self.conv1 = nn.ConvTranspose2d(in_channels=in_c, out_channels=out_c,
                                        kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv2 = nn.Conv2d(in_channels=out_c, out_channels=out_c, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.relu1 = nn.ReLU(True)
        self.relu2 = nn.ReLU(True)
        self.out_img_dim = out_img_dim
        self.out_c = out_c

    def forward(self, x):
        x = self.conv1(x, output_size=(x.size()[0], self.out_c, self.out_img_dim, self.out_img_dim))
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        return x
