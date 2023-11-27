import time

import torch

from ..utils.metrics import Metric, F1Score
from ..utils.base import NotAvailableError
from ..logic.explain import combine_local_explanations, explain_local
from ..utils.relu_nn import get_reduced_model
from .base import BaseClassifier, BaseXModel


class XReluNN(BaseClassifier, BaseXModel):
    """
        Feed forward Neural Network employing ReLU activation function of variable depth but completely interpretable.
        After being trained it provides for local explanation for the prediction on a single example and global
        explanations on the overall dataset

        :param n_classes: int
            number of classes to classify - dimension of the output layer of the network
        :param n_features: int
            number of features - dimension of the input space
        :param hidden_neurons: list
            number of hidden neurons per layer. The length of the list corresponds to the depth of the network.
        :param loss: torch.nn.modules.loss
            type of loss to employ
        :param l1_weight: float
            weight of the l1 regularization on the weights of the network. Allows extracting compact explanations
     """

    def __init__(self, n_classes: int, n_features: int, hidden_neurons: list, loss: torch.nn.modules.loss,
                 dropout_rate: 0.0 = False, l1_weight: float = 1e-4, device: torch.device = torch.device('cpu'),
                 name: str = "relu_net.pth"):

        super().__init__(loss, name, device)
        self.n_classes = n_classes
        self.n_features = n_features

        layers = []
        for i in range(len(hidden_neurons) + 1):
            input_nodes = hidden_neurons[i - 1] if i != 0 else n_features
            output_nodes = hidden_neurons[i] if i != len(hidden_neurons) else n_classes
            layers.append(torch.nn.Linear(input_nodes, output_nodes))
            if i != len(hidden_neurons):
                layers.extend([
                    torch.nn.ReLU(),
                    torch.nn.Dropout(dropout_rate)
                ])
        self.model = torch.nn.Sequential(*layers)
        self.l1_weight = l1_weight

        if n_classes == 1:
            n_classes = 2
        self.explanations = ["" for _ in range(n_classes)]

    def get_loss(self, output: torch.Tensor, target: torch.Tensor, epoch: int = None, epochs: int = None)\
            -> torch.Tensor:
        """
        get_loss method extended from Classifier. The loss passed in the __init__ function of the is employed.
        An L1 weight regularization is also always applied

        :param epochs:
        :param epoch:
        :param output: output tensor from the forward function
        :param target: label tensor
        :return: loss tensor value
        """
        # if epoch is None or epochs is None or epoch > epochs / 2:
        l1_weight = self.l1_weight
        # else:
        #     l1_weight = self.l1_weight * 2 * epoch / epochs
        l1_reg_loss = .0
        for layer in self.model.children():
            if hasattr(layer, "weight"):
                l1_reg_loss += torch.sum(torch.abs(layer.weight))
        output_loss = super().get_loss(output, target)
        return output_loss + l1_weight * l1_reg_loss

    def get_reduced_model(self, x_sample: torch.Tensor) -> torch.nn.Module:
        """
        Get 1-layer model corresponding to the firing path of the model for a specific sample.

        :param x_sample: input sample
        :return: reduced model
        """
        return get_reduced_model(self.model, x_sample)

    def prune(self):
        """
        Prune the inputs of the model.
        """
        # self.model = prune_features(self.model, self.n_classes, self.get_device())
        raise NotAvailableError()

    def get_local_explanation(self, x: torch.Tensor, y: torch.Tensor, x_sample: torch.Tensor,
                              target_class, simplify: bool = True, concept_names: list = None, thr: float = 0.5):
        """
        Get explanation of model decision taken on the input x_sample.

        :param x: input samples
        :param y: target labels
        :param x_sample: input for which the explanation is required
        :param target_class: class ID
        :param simplify: simplify local explanation
        :param concept_names: list containing the names of the input concepts
        :param thr: threshold to use to select important features

        :return: Local Explanation
        """
        return explain_local(self, x, y, x_sample, target_class, method='weights', simplify=simplify, thr=thr,
                             concept_names=concept_names, device=self.get_device(), num_classes=self.n_classes)

    def get_global_explanation(self, x, y, target_class: int, top_k_explanations: int = None,
                               concept_names: list = None, return_time=False, simplify: bool = True,
                               metric: Metric = F1Score(), x_val=None, y_val=None, thr=0.5):
        """
        Generate a global explanation combining local explanations.

        :param y_val:
        :param x_val:
        :param metric:
        :param x: input samples
        :param y: target labels
        :param target_class: class ID
        :param top_k_explanations: number of most common local explanations to combine in a global explanation
                (it controls the complexity of the global explanation)
        :param return_time:
        :param simplify: simplify local explanation
        :param concept_names: list containing the names of the input concepts
        """
        start_time = time.time()
        if isinstance(target_class, torch.Tensor):
            target_class = int(target_class.item())
        if self.explanations[target_class] != "":
            explanation = self.explanations[target_class]
        else:
            explanation, _, _ = combine_local_explanations(self, x, y, target_class, method="weights",
                                                           simplify=simplify, topk_explanations=top_k_explanations,
                                                           concept_names=concept_names, device=self.get_device(),
                                                           num_classes=self.n_classes, metric=metric, x_val=x_val,
                                                           y_val=y_val, thr=thr)
            self.explanations[target_class] = explanation

        elapsed_time = time.time() - start_time
        if return_time:
            return explanation, elapsed_time
        return explanation


if __name__ == "__main__":
    pass
