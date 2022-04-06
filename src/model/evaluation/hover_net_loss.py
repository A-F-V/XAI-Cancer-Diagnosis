from torch import nn, Tensor
import torch
from src.model.evaluation.dice_loss import DiceLoss
from src.model.evaluation.msegradloss import MSEGradLoss


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
        assert len(pred) == len(target)
        co = self.lambdas
        La = nn.MSELoss()(pred[1], target[1])
        Lb = MSEGradLoss(dim=2)(pred[1][:, 0], target[1][:, 0]) + \
            MSEGradLoss(dim=1)(pred[1][:, 1], target[1][:, 1]
                               )  # x=2, y=1, batch=0. Also need to do [:,0] as first dim is batch, second is the map of interest
        Lc = nn.BCELoss()(pred[0].squeeze(), target[0].float().squeeze())
        Ld = DiceLoss()(pred[0].squeeze(), target[0].float().squeeze())
        np_hv_loss = co[0]*La + co[1]*Lb + co[2]*Lc + co[3]*Ld
        if len(pred) == 3:  # !TODO NOT RIGHT! NEED TO CORRECT FOR BACKGROUND AS WELL (ADD THAT DIMENSION IN)
            Le = -6*(target[2].float() * torch.log(pred[2])).mean()  # CROSS ENTROPY LOSS
            Lf = DiceLoss()(pred[2].squeeze(), target[2].float().squeeze())
            return np_hv_loss + co[4]*Le + co[5]*Lf
        return np_hv_loss
