import collections
from abc import ABC, abstractmethod
from statistics import mean
import torch
import numpy as np
from sklearn.metrics import f1_score

from .loss import mutual_information


class Metric(ABC):
    """
    Generic metric interface that needs be extended. It always provides the __call__ method.
    """

    @abstractmethod
    def __call__(self, outputs: torch.Tensor, targets: torch.Tensor) -> float:
        """
        Actual calculation of the metric between between computed outputs and given targets.
        :param outputs: predictions
        :param targets: actual labels
        :return: evaluated metric
        """
        pass


class Accuracy(Metric):
    """
    Accuracy computed between the predictions of the model and the actual labels.
    """

    def __call__(self, outputs: torch.Tensor, targets: torch.Tensor) -> float:
        outputs = outputs if isinstance(outputs, torch.Tensor) else torch.tensor(outputs)
        targets = targets if isinstance(targets, torch.Tensor) else torch.tensor(targets)
        if sum(outputs.shape) > 1:
            outputs, targets = outputs.squeeze(), targets.squeeze()
        if len(outputs.shape) > 1:
            # Multi-Label classification
            if len(targets.shape) > 1:
                outputs = outputs > 0.5
                targets = targets > 0.5
            # Multi-Class classification
            else:
                outputs = outputs.argmax(dim=1)
        else:
            if len(targets.shape) > 1 and targets.sum() == targets.shape[0]:
                targets = targets.argmax(dim=1)
            # Binary classification
            assert targets.shape == outputs.shape, "Target tensor needs to be (N,1) tensor if output is such."
            if outputs.max() <= 1:
                outputs = outputs > 0.5
        if len(outputs.shape) > 1:
            n_samples = targets.shape[0] * targets.shape[1]
        else:
            n_samples = targets.shape[0]
        accuracy = targets.eq(outputs).sum().item() / n_samples * 100
        return accuracy


class TopkAccuracy(Metric):
    """
    Top-k accuracy computed between the predictions of the model and the actual labels.
    It requires to receive an output tensor of the shape (n,c) where c needs to be greater than 1
    :param k: number of elements of the outputs to consider in order to assert a datum as correctly classified
    """

    def __init__(self, k: int = 1):
        self.k = k

    def __call__(self, outputs: torch.Tensor, targets: torch.Tensor) -> float:
        outputs = outputs if isinstance(outputs, torch.Tensor) else torch.tensor(outputs)
        targets = targets if isinstance(targets, torch.Tensor) else torch.tensor(targets)
        assert len(outputs.squeeze().shape) > 1, "TopkAccuracy requires a multi-dimensional outputs"
        assert len(targets.squeeze().shape) == 1, "TopkAccuracy requires a single-dimension targets"
        n_samples = targets.shape[0]
        _, topk_outputs = outputs.topk(self.k, 1)
        topk_acc = topk_outputs.eq(targets.reshape(-1, 1)).sum().item() / n_samples * 100
        return topk_acc


class F1Score(Metric):
    """
    F1 score computed on the predictions of a model and the actual labels. Useful for Multi-label classification.
    """
    def __init__(self, average='macro'):
        self.average = average

    def __call__(self, outputs: torch.Tensor, targets: torch.Tensor) -> float:
        outputs = outputs if isinstance(outputs, torch.Tensor) else torch.tensor(outputs)
        targets = targets if isinstance(targets, torch.Tensor) else torch.tensor(targets)
        assert len(outputs.squeeze().shape) != 1 or len(targets.squeeze().shape) == 1, \
            "Target tensor needs to be (N,1) tensor if output is such."
        # Multi-class
        if len(outputs.squeeze().shape) > 1 and len(targets.squeeze().shape) == 1:
            discrete_output = outputs.argmax(dim=1)
        # Multi-label or Binary classification
        else:
            discrete_output = outputs.cpu().numpy() > 0.5
            targets = targets > 0.5
        targets = targets.cpu().numpy()
        f1_val = f1_score(discrete_output, targets, average=self.average, zero_division=0) * 100
        return f1_val


class MixedMetric(Metric):
    def __init__(self, exclusive_classes_mask: torch.tensor, excl_metric: Metric = Accuracy(),
                 non_excl_metric: Metric = F1Score()):
        super(MixedMetric, self).__init__()
        assert exclusive_classes_mask.dtype == torch.bool, "Only boolean mask are allowed"
        self.exclusive_classes = exclusive_classes_mask
        self.excl_metric = excl_metric
        self.non_excl_metric = non_excl_metric

    def __call__(self, output, target, *args, **kwargs) -> torch.tensor:
        assert output.shape[1] == self.exclusive_classes.squeeze().shape[0], \
            f"boolean mask shape {self.exclusive_classes.squeeze().shape}, " \
            f"different from output number of classes {output.shape[1]}"
        excl_output = output[:, self.exclusive_classes]
        excl_target = target[:, self.exclusive_classes]
        excl_target = excl_target.argmax(dim=1)
        non_excl_output = output[:, ~self.exclusive_classes]
        non_excl_target = target[:, ~self.exclusive_classes]
        excl_metric = self.excl_metric(excl_output, excl_target)
        non_excl_metric = self.non_excl_metric(non_excl_output, non_excl_target)
        return excl_metric + non_excl_metric


class ClusterAccuracy(Metric):
    def __call__(self, outputs: torch.Tensor, targets: torch.Tensor) -> float:
        assert len(targets.squeeze().shape) == 1, "Unsupervised metric require (N, 1) tensor labels"
        assert len(outputs.squeeze().shape) == 2, "Unsupervised metric require (N, C) tensor outputs"
        cluster_accuracies = []
        prevalent_classes = []
        target_counter = np.unique(targets.numpy(), return_counts=True)[1]
        for i in range(outputs.shape[1]):
            cluster_output = outputs[:, i]
            bool_cluster_output = cluster_output > 0.5
            class_cluster_output = targets[bool_cluster_output]
            if class_cluster_output.shape[0] > 1:
                cluster_counter = np.zeros_like(targets.unique())
                for j in range(len(targets.unique())):
                    cluster_counter[j] = (class_cluster_output == j).sum()
                normalized_cluster_counter = cluster_counter/target_counter
                prevalent_class = normalized_cluster_counter.argmax()
                target_class = targets == prevalent_class
                cluster_accuracy = f1_score(bool_cluster_output, target_class) * 100
            else:
                cluster_accuracy = 0.
                prevalent_class = None
            prevalent_classes.append(prevalent_class)
            cluster_accuracies.append(cluster_accuracy)

        mean_accuracy = torch.as_tensor(cluster_accuracies).mean().item()
        return mean_accuracy


        # mi = mutual_information(outputs, normalized=True) * 100
        #
        # return mi.cpu().item()
