from torch import Tensor
from src.utilities.tensor_utilties import reset_ids
import numpy as np
from scipy.stats import mode
from scipy.optimize import linear_sum_assignment
from src.deep_learning.metrics.iou import IoU


def assign_predicted_to_ground_instance_mask(gt: Tensor, pred: Tensor):
    pred = reset_ids(pred.numpy())
    gt = reset_ids(gt.numpy())
    num_gt_cells = gt.max()
    num_pred_cells = pred.max()

    assignment_options = []

    matched = set()

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
    # OLD ALGORITHM WHICH IS GREEDY BUT NOT OPTIMAL
    #  for gt_id, pred_id, iou in options_ranked:
    #     if not(gt_id in gt_matched or pred_id in pred_matched):
    #         TP.add((gt_id, pred_id, iou))
    #         gt_matched.add(gt_id)
    #         pred_matched.add(pred_id)

    # USES Munkres Algorithm

    # 2a) Generate a cost matrix for assignment
    cost_matrix = np.zeros((num_gt_cells, num_pred_cells))
    for gt_id, pred_id, iou in options_ranked:
        cost_matrix[gt_id-1, pred_id-1] = iou  # id == 0 is background
    # 2b) Use Munkres Algorithm to find the optimal assignment and store it in TP
    paired_gt, paired_pred = linear_sum_assignment(cost_matrix, maximize=True)
    for gt, pred, iou in zip(paired_gt, paired_pred, cost_matrix[paired_gt, paired_pred]):
        matched.add((gt+1, pred+1, iou))

    return matched
