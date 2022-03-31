from torch import nn, Tensor


class DiceLoss(nn.Module):
    def __init__(self, epsilon=1e-3):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, pred: Tensor, target: Tensor):
        """Calculates the Dice Loss.

        Args:
            pred (Tensor): Predicted semantic mask. Yi
            target (Tensor): Target semantic mask   Xi

        Returns:
            [float]: The loss value.
        """
        p_flat, t_flat = pred.view(-1), target.view(-1)
        denom = p_flat.sum()+t_flat.sum() + self.epsilon
        num = 2*(p_flat*t_flat).sum() + self.epsilon
        return 1 - num/denom
