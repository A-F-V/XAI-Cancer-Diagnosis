from torch import nn, Tensor


class DiceLoss(nn.Module):
    def __init__(self, epsilon=1e-3):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, pred: Tensor, target: Tensor):
        """Calculates the Dice Loss.

        Args:
            pred (Tensor): Predicted semantic mask. Yi (B x C x H x W)
            target (Tensor): Target semantic mask   Xi (B x C x H x W)

        Returns:
            [float]: The loss value.
        """
        sx, sy, sxy = target.sum(dim=(0, 2, 3)), pred.sum(dim=(0, 2, 3)), (pred*target).sum(dim=(0, 2, 3))
        denom = sx + sy + self.epsilon
        num = 2*sxy + self.epsilon
        each_dice = 1-num/denom
        return each_dice.mean()
