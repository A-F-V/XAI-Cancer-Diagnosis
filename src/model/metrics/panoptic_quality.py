from torch import Tensor
from src.utilities.tensor_utilties import reset_ids
import numpy as np
from scipy.stats import mode


def IoU(img_1: np.ndarray, img_2: np.ndarray):
    """Calculates Intersection over Union.

    Args:
        img_1 (np.ndarray): Image 1
        img_2 (np.ndarray): Image 2

    Returns:
        float: IoU
    """
    p, s = (img_1*img_2).sum(), (img_1+img_2).sum()
    return p/(s-p)


def Panoptic_Quality(pred: Tensor, gt: Tensor):
    """Calculates the panoptic quality of the prediction, as defined in the HoVerNet Paper.

    Args:
        pred (Tensor): Predicted instance segmentation (H,W)
        gt (Tensor): Ground Truth instance segmentation (H,W)

    Returns:
        float: PQ
    """
    # TP = matched, FN = unmatched Ground Truth, FP = unmatched Predicted
    pred = reset_ids(pred.numpy())
    gt = reset_ids(gt.numpy())

    num_gt_cells = gt.max()
    num_pred_cells = pred.max()

    assignment_options = []

    gt_matched = set()
    pred_matched = set()

    TP = set()

    # 1) Find the predicted cell that overlaps the most with the ground truth cell
    for cell_id in range(1, num_gt_cells+1):
        gt_cell_mask = (gt == cell_id)
        mask_on_pred = (pred*gt_cell_mask)
        overlapped_ids = np.unique(mask_on_pred)
        assignment_options += [(cell_id, pred_id, IoU(gt_cell_mask.astype(np.int8),
                                (pred == pred_id).astype(np.int8))) for pred_id in overlapped_ids if pred_id != 0]
        # (gt,pred,IoU)
    options_ranked = sorted(assignment_options, key=lambda triple: triple[2], reverse=True)

    # 2) Assign based on highest IoU (or just overlap?)

    for gt_id, pred_id, iou in options_ranked:
        if not(gt_id in gt_matched or pred_id in pred_matched):
            TP.add((gt_id, pred_id, iou))
            gt_matched.add(gt_id)
            pred_matched.add(pred_id)

    # 3) Collate into matched, unmatched gt, unmatched pred

    FP = set([i for i in range(1, num_pred_cells+1) if i not in pred_matched])
    FN = set([i for i in range(1, num_gt_cells+1) if i not in gt_matched])

    # 4) calculate panoptic quality

    DQ = len(TP)/(len(TP)+len(FP)/2+len(FN)/2)  # Detection Quality
    SQ = sum([assig[2] for assig in TP])/len(TP)  # Segmentation Quality

    return DQ*SQ
