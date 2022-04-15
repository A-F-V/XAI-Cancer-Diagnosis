from torch import Tensor
from src.utilities.tensor_utilties import reset_ids
import numpy as np
from scipy.stats import mode
from scipy.optimize import linear_sum_assignment
from src.algorithm.pair_mask_assignment import assign_predicted_to_ground_instance_mask


def panoptic_quality(pred: Tensor, gt: Tensor):
    """Calculates the panoptic quality of the prediction, as defined in the HoVerNet Paper.

    Args:
        pred (Tensor): Predicted instance segmentation (H,W)
        gt (Tensor): Ground Truth instance segmentation (H,W)

    Returns:
        float: PQ
    """
    # TP = matched,  FP = unmatched Predicted, FN = unmatched Ground Truth,
    pred = reset_ids(pred.numpy())
    gt = reset_ids(gt.numpy())

    TP = assign_predicted_to_ground_instance_mask(gt, pred)

    FP = set([i for i in range(1, pred.max()+1) if i not in set(map(lambda m: m[0], TP))])
    FN = set([i for i in range(1, gt.max()+1) if i not in set(map(lambda m: m[1], TP))])

    DQ = 1 if (len(TP)+len(FP)/2+len(FN)/2) == 0 else len(TP)/(len(TP)+len(FP)/2+len(FN)/2)  # Detection Quality
    SQ = 1 if len(TP) == 0 else sum([assig[2] for assig in TP])/len(TP)  # Segmentation Quality

    return DQ*SQ
