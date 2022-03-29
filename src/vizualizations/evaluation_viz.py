import seaborn as sns
from torch import Tensor


def plot_confusion_matrix(cm: Tensor, class_labels):
    sns.heatmap(cm)
