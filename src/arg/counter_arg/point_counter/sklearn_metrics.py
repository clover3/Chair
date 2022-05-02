from typing import List

from sklearn import metrics


def get_auc(y, pred):
    fpr, tpr, thresholds = metrics.roc_curve(y, pred)
    return metrics.auc(fpr, tpr)


def get_ap(y: List[int], pred: List[float]):
    return metrics.average_precision_score(y, pred)