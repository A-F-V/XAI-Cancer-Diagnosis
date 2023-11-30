import torch
import torch.nn as nn


def wasserstein_distance(y_true, y_pred):
    """
    Compute the Wasserstein Distance between predictions and targets.

    Args:
    y_true (Tensor): The true labels in a one-hot encoded format.
    y_pred (Tensor): The predicted probabilities.

    Returns:
    Tensor: The Wasserstein Distance between the two distributions.
    """
    # Compute the cumulative distribution functions (CDFs)
    cdf_y_true = torch.cumsum(y_true, dim=1)
    cdf_y_pred = torch.cumsum(y_pred, dim=1)

    # Compute the Wasserstein distance as the L1 distance between the CDFs
    wasserstein_dist = torch.mean(torch.abs(cdf_y_true - cdf_y_pred))
    return wasserstein_dist

# Define a custom loss class


class WassersteinLoss(nn.Module):
    def __init__(self):
        super(WassersteinLoss, self).__init__()

    def forward(self, y_pred, y_true):
        return wasserstein_distance(y_true, y_pred)


# Example usage
criterion = WassersteinLoss()
