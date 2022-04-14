from torch import nn, Tensor
import torch
from src.model.evaluation.dice_loss import DiceLoss
from src.model.evaluation.msegradloss import MSEGradLoss


W = torch.as_tensor([9,   47,   22, 50,   33,    1])  # PanNuke Weights
W = W/W.sum()
W = torch.ones(6)


class HoVerNetLoss(nn.Module):
    def __init__(self, lambdas=(1, 2, 1, 1, 1, 1)):  # default is as stipulated in paper
        super().__init__()
        self.lambdas = lambdas
        assert isinstance(lambdas, tuple)

    def forward(self, pred, target):
        """Calculates the HoVerNet Loss.

        Args:
            pred (tuple): Pair of Tensors (semantic_mask, hover_maps) - Either (sm,hv) or (sm,hv,c)
            target (tuple): Pair of Tensors (semantic_mask, hover_maps)

        Returns:
            [float]: The loss value.
        """

        assert len(pred) in [2, 3]
        assert len(target) in [2, 3]
        co = self.lambdas
        La = nn.MSELoss()(pred[1], target[1])
        Lb = MSEGradLoss(dim=2)(pred[1][:, 0], target[1][:, 0]) + \
            MSEGradLoss(dim=1)(pred[1][:, 1], target[1][:, 1]
                               )  # x=2, y=1, batch=0. Also need to do [:,0] as first dim is batch, second is the map of interest
        Lc = nn.BCELoss()(pred[0], target[0].float())
        Ld = DiceLoss()(pred[0], target[0].float())
        np_hv_loss = co[0]*La + co[1]*Lb + co[2]*Lc + co[3]*Ld
        if len(pred) == len(target) == 3:
            Le = -(target[2].float() * torch.log(pred[2])).mean(dim=(0, 2, 3))  # CROSS ENTROPY LOSS
            assert Le.shape == (6,)
            Le = (W.to(Le.device) * Le).sum()
            assert Le >= 0
            Lf = DiceLoss()(pred[2], target[2].float())
            return np_hv_loss + co[4]*Le + co[5]*Lf
        return np_hv_loss
