import torch
from torch import Tensor
from src.utilities.tensor_utilties import reset_ids
from src.model.evaluation.iou import IoU


def AJI(pred: Tensor, gt: Tensor):
    """ Calculates the AJI / Aggregated Jaccard Index
    Args:
        pred (Tensor): Predicted instance segmentation (H,W)
        gt (Tensor): Ground Truth instance segmentation (H,W)

    Returns:
        float: D
    """
    pred = reset_ids(pred.numpy())
    gt = reset_ids(gt.numpy())
    inter_cache, union_cache = {}, {}
    p_sum_cache, q_sum_cache = {}, {}

    unmatched_pred = set(range(1, pred.max()+1))

    matching = {}
    for p in range(1, pred.max()+1):
        for q in range(1, gt.max()+1):
            pmask, qmask = pred == p, gt == q
            ps, qs = p_sum_cache.get(p, pmask.sum()), q_sum_cache.get(q, qmask.sum())

            i = (pmask*qmask).sum()
            u = ps+qs-i

            inter_cache[(p, q)] = i
            union_cache[(p, q)] = u
            p_sum_cache[p] = ps
            q_sum_cache[q] = qs

            if i == 0:
                continue
            unmatched_pred.discard(p)
            if q not in matching:
                matching[q] = (p, i, u)
            elif matching[q][1]/matching[q][2] > i/u:
                matching[q] = (p, i, u)

    cum_inter = sum(map(lambda m: m[1], matching.values()))
    cum_union = sum(map(lambda m: m[2], matching.values()))
    cum_unmatched_sum = sum(map(lambda p: p_sum_cache[p], unmatched_pred))

    assert cum_inter >= 0 and cum_union >= 0 and cum_unmatched_sum >= 0
    if cum_union+cum_unmatched_sum+cum_inter == 0:
        return 1
    return cum_inter/(cum_union+cum_unmatched_sum)
