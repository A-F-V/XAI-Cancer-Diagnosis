import torch
from torch import Tensor
from src.utilities.tensor_utilties import reset_ids


def DICE2(pred: Tensor, gt: Tensor):
    """ Calculates the DICE2 / Aggregated DICE coefficient
    Args:
        pred (Tensor): Predicted instance segmentation (H,W)
        gt (Tensor): Ground Truth instance segmentation (H,W)

    Returns:
        float: D
    """
    intersection, union = 0, 0
    pred = reset_ids(pred.numpy())
    gt = reset_ids(gt.numpy())
    for p in range(1, pred.max()+1):
        for q in range(1, gt.max()+1):
            pmask, qmask = pred == p, gt == q
            i = (pmask*qmask).sum()
            if i > 0:
                intersection += i
                union += pmask.sum() + qmask.sum()
    if union == 0:
        return 1
    return 2*intersection/union
