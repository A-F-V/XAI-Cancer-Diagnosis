import torch
from torch import nn, Tensor
from torch.nn.functional import interpolate


class ResidualUnit(nn.Module):  # ONLY RESNET 50 for now #todo extend to all resnets (small for quick testing)
    """Residual block as defined in:
    He, Kaiming, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. "Deep residual learning
    for image recognition." In Proceedings of the IEEE conference on computer vision
    and pattern recognition, pp. 770-778. 2016.
    """

    def __init__(self, in_features, channels, kernels, stride):
        """Creates the Residual Unit.

        Args:
            in_features (int): Number of input channels
            channels (list of int): The number of feature maps at each level
            kernels (list of int): The kernel size at each level
            stride (int): The stride at the top level
        """
        super(ResidualUnit, self).__init__()
        layers = []
        cur_channels = in_features

        layers.append(nn.BatchNorm2d(in_features))
        layers.append(nn.ReLU(inplace=True))
        for i, (out_features, kernel) in enumerate(zip(channels, kernels)):
            layers.append(nn.Conv2d(cur_channels, out_features, kernel_size=kernel, stride=stride if i ==
                          0 else 1, padding=(kernel-1)//2))  # hopefully keeps image size same between downsamples
            cur_channels = out_features
            layers.append(nn.BatchNorm2d(cur_channels))
            layers.append(nn.ReLU(inplace=True))

        self.layers = nn.Sequential(*layers)

        self.shortcut = nn.Conv2d(in_features, cur_channels, kernel_size=1,
                                  stride=1) if in_features != cur_channels else None  # SLIGHTLY DIFFERENT TO ORIGINAL

    def forward(self, sample: Tensor):
        contribution = self.layers(sample)
        short = sample if self.shortcut == None else self.shortcut(sample)
        # need special shortcut layer because there may be an imbalance of channels or stride
        return interpolate(short, contribution.size()[2:]) + contribution  # as size is (batch, channel, height, width)
