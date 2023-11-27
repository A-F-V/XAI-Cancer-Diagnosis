import collections
from typing import List

import numpy as np
import torch
from sympy import simplify_logic, to_dnf
from torch.nn import ModuleList

from .eval import replace_names, test_explanation, simplify_formula
from .psi_nn import _build_truth_table
from ..utils.base import collect_parameters, to_categorical
from ..utils.metrics import F1Score, Metric
from ..utils.selection import rank_pruning, rank_weights, rank_lime


def combine_local_explanations(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor,
                               target_class: int, method: str, simplify: bool = True,
                               topk_explanations: int = None, concept_names: List = None,
                               device: torch.device = torch.device('cpu'), num_classes: int = None,
                               return_accuracy: bool = False, metric: Metric = F1Score(),
                               x_val: torch.tensor = None, y_val = None, thr=0.5):
    """
    Generate a global explanation combining local explanations.

    :param y_val:
    :param x_val:
    :param metric:
    :param model: pytorch model
    :param x: input samples
    :param y: target labels
    :param target_class: class ID
    :param method: local feature importance method
    :param simplify: simplify local explanation
    :param topk_explanations: number of most common local explanations to combine in a global explanation (it controls
            the complexity of the global explanation)
    :param concept_names: list containing the names of the input concepts
    :param device: cpu or cuda device
    :param num_classes: override the number of classes
    :param return_accuracy: whether to return also the accuracy of the explanations or not
    :param thr: threshold to select important features

    :return: Global explanation, predictions, and ranking of local explanations
    """
    assert topk_explanations is not None or (x_val is not None and y_val is not None), \
        "validation data need to be passed when the number of top explanations to retain is not specified"

    x, y = x.to(device), y.to(device)
    y = to_categorical(y).squeeze()
    if x_val is not None and y_val is not None:
        x_val, y_val = x_val.to(device), y_val.to(device)
        y_val = to_categorical(y_val).squeeze()
    assert (y == target_class).any(), "Cannot get explanation if target class is not amongst target labels"

    # # collapse samples having the same boolean values and class label different from the target class
    if hasattr(model, "model"):
        target_model = model.model
        if isinstance(target_model, ModuleList):
            if target_class == 1 and len(target_model) == 1:  # binary classification case
                target_model = target_model[0]
            else:
                target_model = target_model[target_class]
    else:
        target_model = model
    w, b = collect_parameters(target_model, device)
    feature_weights = w[0]
    feature_used_bool = np.sum(np.abs(feature_weights), axis=0) > 0
    feature_used = np.sort(np.nonzero(feature_used_bool)[0])
    _, idx = np.unique((x[:, feature_used][y == target_class] >= 0.5).cpu().detach().numpy(), axis=0, return_index=True)
    # _, idx = np.unique((x[y == target_class] >= 0.5).cpu().detach().numpy(), axis=0, return_index=True)
    if topk_explanations is None:
        x_target = x[y == target_class][idx]
        y_target = y[y == target_class][idx]
    else:
        x_target = x[y == target_class]
        y_target = y[y == target_class]
    # print(len(y_target))

    # get model's predictions
    preds = model(x_target)
    preds = to_categorical(preds)

    # identify samples correctly classified of the target class
    correct_mask = y_target.eq(preds)
    x_target_correct = x_target[correct_mask]
    y_target_correct = y_target[correct_mask]

    # collapse samples having the same boolean values and class label different from the target class
    _, idx = np.unique((x[:, feature_used][y != target_class] > 0.5).cpu().detach().numpy(), axis=0, return_index=True)
    if topk_explanations is None:
        x_reduced_opposite = x[y != target_class][idx]
        y_reduced_opposite = y[y != target_class][idx]
    else:
        x_reduced_opposite = x[y != target_class]
        y_reduced_opposite = y[y != target_class]
    preds_opposite = model(x_reduced_opposite)
    if len(preds_opposite.squeeze(-1).shape) > 1:
        preds_opposite = torch.argmax(preds_opposite, dim=1)
    else:
        preds_opposite = (preds_opposite > 0.5).squeeze()

    # identify samples correctly classified of the opposite class
    correct_mask = y_reduced_opposite.eq(preds_opposite)
    x_reduced_opposite_correct = x_reduced_opposite[correct_mask]
    y_reduced_opposite_correct = y_reduced_opposite[correct_mask]

    # select the subset of samples belonging to the target class
    x_validation = torch.cat([x_reduced_opposite_correct, x_target_correct], dim=0)
    y_validation = torch.cat([y_reduced_opposite_correct, y_target_correct], dim=0)

    # generate local explanation only for samples where:
    # 1) the model's prediction is correct and
    # 2) the class label corresponds to the target class
    local_explanations = []
    local_explanations_raw = {}
    local_explanations_translated = []
    for sample_id, (xi, yi) in enumerate(zip(x_target_correct, y_target_correct)):
        local_explanation_raw = explain_local(model, x_validation, y_validation,
                                              xi.to(torch.float), target_class,
                                              method=method, simplify=False,
                                              concept_names=None, device=device,
                                              num_classes=num_classes, thr=thr)

        if local_explanation_raw in ['', 'False', 'True']:
            continue

        if local_explanation_raw in local_explanations_raw:
            local_explanation = local_explanations_raw[local_explanation_raw]
        # elif simplify:
        #     local_explanation = simplify_formula(local_explanation_raw, model,
        #                                          x_validation, y_validation,
        #                                          xi, target_class)
        else:
            local_explanation = local_explanation_raw

        if local_explanation in ['']:
            continue

        local_explanations_raw[local_explanation_raw] = local_explanation
        local_explanations.append(local_explanation)

        # get explanations using original concept names (if any)
        if concept_names is not None:
            local_explanation_translated = replace_names(local_explanation, concept_names)
        else:
            local_explanation_translated = local_explanation
        local_explanations_translated.append(local_explanation_translated)

    if len(local_explanations) == 0:
        if not return_accuracy:
            return '', np.array, collections.Counter()
        else:
            return '', np.array, collections.Counter(), 0.

    # get most frequent local explanations
    counter = collections.Counter(local_explanations)
    counter_translated = collections.Counter(local_explanations_translated)

    if topk_explanations is None or len(counter) < topk_explanations:
        topk_explanations = len(counter)
    most_common_explanations = []
    best_accuracy = 0
    for i, (explanation, _) in enumerate(counter.most_common(topk_explanations)):
        most_common_explanations.append(explanation)
        global_explanation = ' | '.join(most_common_explanations)
        if x_val is not None and y_val is not None:
            accuracy, predictions = test_explanation(global_explanation, target_class,
                                                     x_val, y_val, metric=metric)
            if accuracy <= best_accuracy:
                most_common_explanations.remove(explanation)
            else:
                best_accuracy = accuracy

    # the global explanation is the disjunction of local explanations
    global_explanation = ' | '.join(most_common_explanations)
    # avoid simplify if number of terms too long
    if simplify and len(np.unique(np.asarray(global_explanation.split(" ")))) < 20 + 2:
        global_explanation_simplified_str = str(simplify_logic(global_explanation, 'dnf', force=simplify))
    else:
        global_explanation_simplified_str = str(to_dnf(global_explanation))
    if not global_explanation_simplified_str:
        if not return_accuracy:
            return '', np.array, collections.Counter()
        else:
            return '', np.array, collections.Counter(), 0.

    # predictions based on FOL formula
    accuracy, predictions = test_explanation(global_explanation_simplified_str, target_class,
                                             x_validation, y_validation, metric=metric)

    # replace concept names
    if concept_names is not None:
        global_explanation_simplified_str = replace_names(global_explanation_simplified_str, concept_names)

    if not return_accuracy:
        return global_explanation_simplified_str, predictions, counter_translated
    else:
        return global_explanation_simplified_str, predictions, counter_translated, accuracy


def explain_local(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor, x_sample: torch.Tensor,
                  target_class: int, method: str, simplify: bool = True, concept_names: List = None,
                  device: torch.device = torch.device('cpu'), num_classes: int = None, thr=0.5) -> str:
    """
    Generate a local explanation for a single sample.

    :param model: pytorch model
    :param x: input samples
    :param y: target labels
    :param x_sample: input for which the explanation is required
    :param target_class: class ID
    :param method: local feature importance method
    :param simplify: simplify local explanation
    :param concept_names: list containing the names of the input concepts
    :param device: cpu or cuda device
    :param num_classes: override the number of classes
    :param thr: threshold to select important features
    :return: Local explanation
    """
    if len(x_sample.shape) == 1:
        x_sample = x_sample.unsqueeze(0)

    if hasattr(model, 'model'):
        model_to_rank = model.model
        if isinstance(model_to_rank, torch.nn.ModuleList):
            model_to_rank = model_to_rank[target_class] if model.n_classes > 1 else model_to_rank[0]
    else:
        model_to_rank = model

    if method == 'pruning':
        feature_used = rank_pruning(model_to_rank, x_sample, y, device, num_classes=num_classes)

    elif method == 'weights':
        feature_used = rank_weights(model_to_rank, x_sample, device, thr=thr)

    elif method == 'lime':
        feature_used = rank_lime(model_to_rank, x, x_sample, 4, device)

    # elif method == 'shap':
    #     pass

    else:
        feature_used = rank_weights(model, x_sample, device)

    # explanation is the conjunction of non-pruned features
    explanation = ''
    for j in feature_used:
        if explanation:
            explanation += ' & '
        explanation += f'feature{j:010}' if x_sample[:, j] > 0.5 else f'~feature{j:010}'

    if simplify:
        explanation = simplify_formula(explanation, model, x, y, x_sample, target_class)

    # replace concept placeholders with concept names
    if concept_names is not None:
        explanation = replace_names(explanation, concept_names)

    return explanation


def explain_global(model: torch.nn.Module, n_classes: int,
                   target_class: int, concept_names: list = None,
                   device: torch.device = torch.device('cpu')) -> str:
    """
    Explain the behavior of the model for a whole class.

    :param model: torch model
    :param n_classes: number of classes
    :param target_class: target class
    :param concept_names: list containing the names of the input concepts
    :param device: cpu or cuda device
    :return: Global explanation for a single class
    """
    # identify non pruned features
    w, b = collect_parameters(model, device)
    feature_weights = w[0]
    block_size = feature_weights.shape[0] // n_classes
    feature_used_bool = np.sum(np.abs(feature_weights[target_class * block_size : (target_class + 1) * block_size]),
                               axis=0) > 0
    feature_used = np.nonzero(feature_used_bool)[0]

    # if the number of features is too high, then don't even try to get something
    if len(feature_used) > 20:
        return 'The formula is too complex!'

    # build truth table and use it to query the model
    truth_table = _build_truth_table(len(feature_used))
    truth_table_tensor = torch.FloatTensor(truth_table).to(device)
    input_table = torch.zeros((len(truth_table), feature_weights.shape[1])).to(device)
    input_table[:, feature_used] = truth_table_tensor
    predictions = model(input_table)
    if device.type != 'cpu':
        predictions = predictions.cpu()
    predictions = predictions.detach().numpy().squeeze()

    if len(predictions.shape) > 1:
        predictions = np.argmax(predictions, axis=1) == target_class
    else:
        predictions = predictions > 0.5

    # select the rows of the input truth table for which the output is true
    X = truth_table[np.nonzero(predictions)]

    # if the output is never true, then return false
    if np.shape(X)[0] == 0: return "False"

    # if the output is never false, then return true
    if np.shape(X)[0] == np.shape(truth_table)[0]: return "True"

    # compute the formula
    formula = ''
    n_rows, n_features = X.shape
    for i in range(n_rows):
        # if the formula is not empty, start appending an additional term
        if formula != '':
            formula = formula + "|"

        # open the bracket
        formula = formula + "("
        for j in range(n_features):
            # get the name (column index) of the feature
            feature_name = f'feature{feature_used[j]:010}'

            # if the feature is not active,
            # then the corresponding predicate is false,
            # then we need to negate the feature
            if X[i][j] == 0:
                formula += "~"

            # append the feature name
            formula += feature_name + "&"

        formula = formula[:-1] + ')'

    # replace "not True" with "False" and vice versa
    formula = formula.replace('~(True)', 'False')
    formula = formula.replace('~(False)', 'True')

    # simplify formula
    simplified_formula = simplify_logic(formula, 'dnf', force=True)
    simplified_formula = str(simplified_formula)

    # replace concept names
    if concept_names is not None:
        simplified_formula = replace_names(str(simplified_formula), concept_names)

    return simplified_formula
