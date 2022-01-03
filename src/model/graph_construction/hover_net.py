from torch import nn
import numpy as np
from src.model.components.residual_unit import ResidualUnit

resnet_sizes = [18, 34, 50, 101, 152]


class HoVerNet(nn.Module):
    def __init__(self, resnet_size):
        super(HoVerNet, self).__init__()
        self.encoder = HoVerNetEncoder(resnet_size)
        self.np_branch = HoVerNetDecoder()
        self.hover_branch = HoVerNetDecoder()

    def forward(self, sample):
        latent = self.encoder(sample)
        semantic_mask = self.np_branch(latent)
        hover_maps = self.hover_branch(latent)
        return semantic_mask, hover_maps


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
    input_channel_lookup = [64, 256, 512, 1024, 2048]

    kernels, channels = None, None
    stride = 1 if depth == 2 else 2
    if resnet_size < 50:
        kernels = [3, 3]
        channels = (np.array([64, 64])*(2**(depth-2))).tolist()
    else:
        kernels = [1, 3, 1]
        channels = (np.array([64, 64, 256])*(2**(depth-2))).tolist()
    times = times_lookup[resnet_size][depth-2]
    in_channel = input_channel_lookup[depth-2]
    out_channel = input_channel_lookup[depth-1]

    return nn.Sequential(ResidualUnit(in_channel, channels, kernels, stride), *[ResidualUnit(out_channel, channels, kernels, 1) for _ in range(times-1)])


class HoVerNetEncoder(nn.Module):
    def __init__(self, resnet_size):
        super(HoVerNetEncoder, self).__init__()
        self.resnet_size = resnet_size

        self.layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            *[create_resnet_conv_layer(self.resnet_size, depth) for depth in range(2, 6)],
            nn.Conv2d(2048, 1024, kernel_size=1, stride=1, padding=0, bias=False),
        )


class HoVerNetDecoder(nn.Module):
    def __init__(self):
        super(HoVerNetDecoder, self).__init__()
