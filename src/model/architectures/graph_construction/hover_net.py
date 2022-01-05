from torch import nn, optim
import numpy as np
from src.model.architectures.components.residual_unit import ResidualUnit
from src.model.architectures.components.dense_decoder_unit import DenseDecoderUnit
import pytorch_lightning as pl
from src.model.metrics.hover_net_loss import HoVerNetLoss

resnet_sizes = [18, 34, 50, 101, 152]

# todo consider using ModuleList instead of Sequential?


class HoVerNet(pl.LightningModule):
    """HoVerNet Architecture (without the Nuclei Classification Branch) as described in:
        Graham, Simon, et al. "Hover-net: Simultaneous segmentation and classification of nuclei in multi-tissue histology images." Medical Image Analysis 58 (2019): 101563.

    """

    def __init__(self, num_batches=0, train_loader=None, val_loader=None, **kwargs):
        super(HoVerNet, self).__init__()
        resnet_size = kwargs["RESNET_SIZE"]
        assert resnet_size in resnet_sizes
        decodersize = (resnet_sizes.index(resnet_size)+2)*0.25
        self.encoder = HoVerNetEncoder(resnet_size)
        self.np_branch = nn.Sequential(HoVerNetDecoder(decodersize), HoVerNetBranchHead("np"))
        self.hover_branch = nn.Sequential(HoVerNetDecoder(decodersize), HoVerNetBranchHead("hover"))
        self.learning_rate = kwargs["START_LR"]
        self.args = kwargs
        self.num_batches = num_batches
        self.train_loader = train_loader
        self.val_loader = val_loader

   # def setup(self, stage=None):
   #     self.logger.experiment.log_params(self.args)

    def forward(self, sample):
        latent = self.encoder(sample)
        semantic_mask = self.np_branch(latent)
        hover_maps = self.hover_branch(latent)
        return semantic_mask, hover_maps

    def predict(self, sample):
        pipeline = nn.Sequential(self.encoder.eval(), self.np_branch.eval())
        return pipeline(sample) > 0.5

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        if self.args["ONE_CYCLE"]:
            lr_scheduler = optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=self.args['MAX_LR'], total_steps=self.num_batches,  three_phase=True)
            return [optimizer], [{'scheduler': lr_scheduler, 'interval': 'step'}]
        else:
            return optimizer

    def training_step(self, train_batch, batch_idx):
        i, sm, hv = train_batch['image'].float(), train_batch['semantic_mask'].float(), train_batch['hover_map'].float()

        y = (sm, hv)
        y_hat = self(i)

        loss = HoVerNetLoss()(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        i, sm, hv = val_batch['image'].float(), val_batch['semantic_mask'].float(), val_batch['hover_map'].float()

        y = (sm, hv)
        y_hat = self(i)

        loss = HoVerNetLoss()(y_hat, y)
        self.log("val_loss", loss)
        return loss

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader


def create_resnet_conv_layer(resnet_size, depth):
    """Creates the conv layer for the resnet. Specified in https://iq.opengenus.org/content/images/2020/03/Screenshot-from-2020-03-20-15-49-54.png

    Args:
        resnet_size (int): the canonincal resnet layer (18, 34, 50, 101, 152)
        depth (iny): what layer number (2,3,4,5)

    Returns:
        nn.Sequential : The pytorch layer
    """
    assert resnet_size in resnet_sizes
    assert depth >= 2 and depth <= 5

    times_lookup = {18: [2, 2, 2, 2], 34: [3, 4, 6, 3], 50: [3, 4, 6, 3], 101: [3, 4, 23, 3], 152: [3, 8, 36, 3]}
    first_channel_lookup = [64, 128, 256, 512]

    kernels, channels = None, None
    stride = 1 if depth == 2 else 2
    if resnet_size < 50:
        kernels = [3, 3]
        channels = (np.array([64, 64])*(2**(depth-2))).tolist()
    else:
        kernels = [1, 3, 1]
        channels = (np.array([64, 64, 256])*(2**(depth-2))).tolist()
    times = times_lookup[resnet_size][depth-2]
    in_channel = 64 if depth == 2 else (first_channel_lookup[depth-3]*(4 if resnet_size >= 50 else 1))
    out_channel = first_channel_lookup[depth-2]*4 if resnet_size >= 50 else first_channel_lookup[depth-2]

    return nn.Sequential(ResidualUnit(in_channel, channels, kernels, stride), *[ResidualUnit(out_channel, channels, kernels, 1) for _ in range(times-1)])


class HoVerNetEncoder(nn.Module):  # Returns 1024 maps with images down sampled by 8x
    def __init__(self, resnet_size):
        super(HoVerNetEncoder, self).__init__()
        self.resnet_size = resnet_size

        self.layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            *[create_resnet_conv_layer(self.resnet_size, depth) for depth in range(2, 6)],
            nn.Conv2d(2048 if resnet_size >= 50 else 512, 1024, kernel_size=1, stride=1, padding=0, bias=False),
        )

    def forward(self, sample):
        return self.layers(sample)


def create_dense_decoder_blocks(in_channels, times):  # todo
    # because of concat in DenseDecoderUnit, we add 32 channels to the input
    return nn.Sequential(*[DenseDecoderUnit(in_channels+i*32) for i in range(times)])


class HoVerNetDecoder(nn.Module):
    def __init__(self, size):
        super(HoVerNetDecoder, self).__init__()
        self.size = size
        units1, units2 = int(8*size), int(4*size)  # can choose how large you want decoder to be

        self.level1 = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=5, padding=2),
            create_dense_decoder_blocks(256, units1),
            nn.Conv2d(256+units1*32, 512, kernel_size=1, padding=0),
        )
        self.level2 = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=5, padding=2),
            create_dense_decoder_blocks(128, units2),
            nn.Conv2d(128+32*units2, 128, kernel_size=1, padding=0),
        )
        self.level3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=5,
                      padding=2, stride=1, dilation=1, bias=False),
            nn.Conv2d(256, 64, kernel_size=1,
                      padding=0, stride=1, dilation=1, bias=False))

    def forward(self, sample):
        out = nn.Upsample(scale_factor=2, mode='nearest')(sample)
        out = self.level1(out)
        out = nn.Upsample(scale_factor=2, mode='nearest')(out)
        out = self.level2(out)
        out = nn.Upsample(scale_factor=2, mode='nearest')(out)
        return self.level3(out)


class HoVerNetBranchHead(nn.Module):
    def __init__(self, branch):
        super(HoVerNetBranchHead, self).__init__()
        assert branch in ["np", "hover"]

        self.activate = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        if branch == "np":
            self.head = nn.Sequential(
                nn.Conv2d(64, 1, kernel_size=1, padding=0, bias=False),
                nn.Sigmoid())
        else:  # todo is there a better activation function?
            self.head = nn.Sequential(
                nn.Conv2d(64, 2, kernel_size=1, padding=0, bias=False))

    def forward(self, sample):
        return self.head(self.activate(sample))
