from torch import nn, Tensor

from src.model.metrics.dice_loss import DiceLoss
from src.model.metrics.msegradloss import MSEGradLoss


class HoVerNetLoss(nn.Module):
    def __init__(self, lambdas=(1, 2, 1, 1)):  # default is as stipulated in paper
        super().__init__()
        self.lambdas = lambdas
        assert isinstance(lambdas, tuple)

    def forward(self, pred, target):
        """Calculates the HoVerNet Loss.

        Args:
            pred (tuple): Pair of Tensors (semantic_mask, hover_maps)
            target (tuple): Pair of Tensors (semantic_mask, hover_maps)

        Returns:
            [float]: The loss value.
        """
        La = nn.MSELoss()(pred[1], target[1])
        Lb = MSEGradLoss(dim=1)(pred[0], target[0])+MSEGradLoss(dim=0)(pred[0], target[0])  # x=1, y=0
        Lc = nn.CrossEntropyLoss()(pred[0], target[0])
        Ld = DiceLoss()(pred[0], target[0])
        co = self.lambdas
        return co[0]*La + co[1]*Lb + co[2]*Lc + co[3]*Ld
