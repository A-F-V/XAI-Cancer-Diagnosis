import torch
from torch import Tensor


def confusion_matrix(ground_truth: Tensor, predicted: Tensor, num_classes: int):
    """
    Computes the confusion matrix for a set of predictions and ground truths.
    :param ground_truth: The ground truth tensor.
    :param predicted: The predicted tensor.
    :param num_classes: The number of classes.
    :return: The confusion matrix.
    """
    # Compute the confusion matrix.
    confusion_matrix = torch.zeros(num_classes, num_classes, dtype=torch.int64)
    for i in range(ground_truth.size(0)):
        confusion_matrix[ground_truth[i], predicted[i]] += 1

    return confusion_matrix
