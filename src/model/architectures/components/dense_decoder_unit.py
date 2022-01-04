import torch
from torch import nn


class DenseDecoderUnit(nn.Module):
    def __init__(self, input_features):
        super(DenseDecoderUnit, self).__init__()
        self.layers = nn.Sequential(
            nn.BatchNorm2d(input_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(input_features, 128, kernel_size=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 32, kernel_size=5, padding=2),
        )

    def forward(self, sample):  # todo check that contribution isn't changing data
        contribution = self.layers(sample)
        return torch.cat([sample, contribution])  # todo use interpolation?
