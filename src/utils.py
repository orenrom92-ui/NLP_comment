import random
import numpy as np
import torch

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score
)


def conf_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def func_metrics_multiclass(preds, labels):
    return {
        'accuracy': accuracy_score(labels, preds),
        'f1': f1_score(labels, preds, average='macro', zero_division=0),
        'precision': precision_score(labels, preds, average='macro', zero_division=0),
        'recall': recall_score(labels, preds, average='macro', zero_division=0),
    }