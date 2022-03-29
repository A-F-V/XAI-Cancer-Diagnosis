from torch import nn, Tensor
from src.utilities.tensor_utilties import gradient


class MSEGradLoss(nn.Module):
    def __init__(self, dim=0):
        """

        Args:
            dim (int, optional): X grad = 0, Y grad = 1. Defaults to X.
        """
        super().__init__()
        self.dim = dim

    def forward(self, pred: Tensor, target: Tensor):
        """Calculates the MSE of the Gradients Loss.

        Args:
            pred (Tensor): 2D tensor (possibly batch of) 
            target (Tensor): 2D tensor (possibly batch of) 

        Returns:
            [float]: The loss value.
        """
        pred_grad = gradient(pred, self.dim)
        target_grad = gradient(target, self.dim)
        return nn.MSELoss()(pred_grad, target_grad)
