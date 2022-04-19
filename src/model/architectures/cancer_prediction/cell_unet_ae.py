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


class UNET_AE(pl.LightningModule):
    def __init__(self, predictOnly=False, data_set_path=None, img_size=64, num_steps=0, **kwargs):
        super(UNET_AE, self).__init__()
        self.args = kwargs
        self.img_size = img_size
        self.learning_rate = kwargs["START_LR"] if "START_LR" in kwargs else 1e-3
        self.num_steps = num_steps
        #self.data_set = BACH_Cells(data_set_path, img_size=self.img_size, **kwargs)
        # self.data_loader = DataLoader(
        #    self.data_set, batch_size=kwargs["BATCH_SIZE"], shuffle=True, num_workers=kwargs["NUM_WORKERS"])

        #self.encoder = UNET_Encoder(in_channels=3, out_channels=64, num_steps=self.num_steps, **kwargs)
        #self.decoder = UNET_Decoder(in_channels=64, out_channels=3, num_steps=self.num_steps, **kwargs)

        self.src_folder = data_set_path
        self.num_steps = num_steps
        self.width = kwargs["WIDTH"] if "WIDTH" in kwargs else 32
        self.dropout = kwargs["DROPOUT"] if "DROPOUT" in kwargs else 0

        self.encoder('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_resnet50', pretrained=True)
        self.predictor = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(self.dropout),
            nn.Linear((512*self.width*16)//32, (1000*self.width)//32),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm1d((1000*self.width)//32),
            nn.Dropout(self.dropout),
            nn.Linear((1000*self.width)//32, (200*self.width)//32),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm1d((200*self.width)//32),
            nn.Dropout(self.dropout),
            nn.Linear((200*self.width)//32, (40*self.width)//32),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm1d((40*self.width)//32),
            nn.Linear((40*self.width)//32, 4),
            nn.Softmax(dim=1))

        if data_set_path != None:
            self.setup_datasets()

    def setup_datasets(self):
        tr_trans = Compose([                                       # ASPIRATIONAL
            # , RandomChoice(transforms=[GaussianBlur(kernel_size=3), AddGaussianNoise(0, 0.01)], p=[0.5, 0.5])]
            RandomHorizontalFlip(), RandomVerticalFlip(), ColorJitter(
                brightness=self.args["IMG_AUG"], contrast=self.args["IMG_AUG"], saturation=self.args["IMG_AUG"], hue=(-self.args["IMG_AUG"]/2, self.args["IMG_AUG"]/2))
        ])
        val_trans = Compose([])

        train_set_size, val_set_size = self.args["NUM_BATCHES_PER_EPOCH"] * \
            self.args["BATCH_SIZE_TRAIN"], self.args["NUM_BATCHES_PER_EPOCH"]*self.args["BATCH_SIZE_VAL"]

        # train_set, val_set = train_val_split(BACH_Cells, src_folder, 0.8, tr_trans=tr_trans, val_trans=val_trans)
        train_set, val_set = BACH_Cells(self.src_folder, transform=tr_trans, val=False), BACH_Cells(
            self.src_folder, transform=val_trans, val=True)

        if self.args["EPOCH_MODE"] != "FULL":
            train_set, val_set = random_subset(train_set, train_set_size), random_subset(val_set, val_set_size)

        self.train_loader = DataLoader(train_set, batch_size=self.args["BATCH_SIZE_TRAIN"],
                                       shuffle=True, num_workers=self.args["NUM_WORKERS"],
                                       persistent_workers=self.args["EPOCH_MODE"] == "FULL"
                                       )
        self.val_loader = DataLoader(val_set, batch_size=self.args["BATCH_SIZE_VAL"],
                                     shuffle=False, num_workers=self.args["NUM_WORKERS"],
                                     persistent_workers=self.args["EPOCH_MODE"] == "FULL")

    def forward(self, x):
        enc1 = self.unet.encoder1(x)
        enc2 = self.unet.encoder2(self.unet.pool1(enc1))
        enc3 = self.unet.encoder3(self.unet.pool2(enc2))
        enc4 = self.unet.encoder4(self.unet.pool3(enc3))

        bottleneck = self.unet.bottleneck(self.unet.pool4(enc4))

        dec4 = self.unet.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.unet.decoder4(dec4)
        dec3 = self.unet.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.unet.decoder3(dec3)
        dec2 = self.unet.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.unet.decoder2(dec2)
        dec1 = self.unet.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.unet.decoder1(dec1)

        x_hat = self.unet.conv(dec1).clip(0, 1)
        y_hat = self.predictor(bottleneck)

        return x_hat, y_hat

    @incremental_forward(64)
    def forward_pred(self, x):
        return self.forward(x)[1]

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
        cells, y = train_batch["img"], categorise(train_batch["diagnosis"].float())
        cell_hat, y_hat = self.forward(cells)
        batch_size = y.shape[0]

        mse, ce = F.mse_loss(cell_hat, cells), F.nll_loss(torch.log(y_hat), y)*10
        loss = ce + mse

        pred_cat = y_hat.argmax(dim=1)
        canc_pred = (torch.where(pred_cat.eq(0) | pred_cat.eq(3), 0, 1)).float()
        canc_grd = (torch.where(y.eq(0) | y.eq(3), 0, 1)).float()
        acc = (pred_cat == y).float().mean()
        canc_acc = (canc_pred == canc_grd).float().mean()
        self.log("train_acc", acc)
        self.log("train_canc_acc", canc_acc)
        self.log("train_mse", mse)
        self.log("train_ce", ce)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        cells, y = val_batch['img'], categorise(val_batch["diagnosis"].float())
        cell_hat, y_hat = self.forward(cells)
        batch_size = y.shape[0]
        mse, ce = F.mse_loss(cell_hat, cells), F.nll_loss(torch.log(y_hat), y)*10
        loss = mse+ce

        pred_cat = y_hat.argmax(dim=1)
        canc_pred = (torch.where(pred_cat.eq(0) | pred_cat.eq(3), 0, 1)).float()
        canc_grd = (torch.where(y.eq(0) | y.eq(3), 0, 1)).float()
        acc = (pred_cat == y).float().mean()
        canc_acc = (canc_pred == canc_grd).float().mean()
        self.log("val_loss", loss)
        self.log("val_mse", mse)
        self.log("val_ce", ce)
        self.log("val_acc", acc)
        self.log("val_canc_acc", canc_acc)
        return loss

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def on_validation_epoch_end(self):
        # if self.current_epoch != 0:
        sample = self.val_dataloader().dataset[0]['img'].unsqueeze(0).to(self.device)
        pred_out = self.forward(sample)[0]
        f = plot_images([tensor_to_numpy(sample.squeeze().detach().cpu()),
                        tensor_to_numpy(pred_out.squeeze().detach().cpu())], (2, 1))
        log_plot(plt=f, name=f"{self.current_epoch}", logger=self.logger.experiment, run_id=self.logger.run_id)
        # self.logger.experiment.log_artifact(local_path=gif_diag_path, artifact_path=f"Cell_Seg_{self.current_epoch}", run_id=self.logger.run_id)  # , "sliding_window_gif")

    def on_train_epoch_start(self):
        if self.args["EPOCH_MODE"] != "FULL":
            self.setup_datasets()


def categorise(t: Tensor):
    if len(t.shape) == 1:
        t = t.unsqueeze(0)
    return t.argmax(dim=1)
