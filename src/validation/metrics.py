"""
Evaluation metrics for marine debris detection.

Placeholder module: extend with IoU, per-class precision/recall/F1,
confusion matrix utilities, and segmentation-specific metrics as the
project develops.
"""

import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score


def compute_iou(y_true: np.ndarray, y_pred: np.ndarray, positive_class: int = 1) -> float:
    """
    Intersection over Union for a binary class.

    IoU = TP / (TP + FP + FN)
    """
    y_true_bin = (y_true == positive_class)
    y_pred_bin = (y_pred == positive_class)

    intersection = np.logical_and(y_true_bin, y_pred_bin).sum()
    union = np.logical_or(y_true_bin, y_pred_bin).sum()

    if union == 0:
        return float("nan")
    return float(intersection / union)


def compute_binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Return precision, recall, F1, IoU for the positive class (1)."""
    return {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall":    recall_score(y_true, y_pred, zero_division=0),
        "f1":        f1_score(y_true, y_pred, zero_division=0),
        "iou":       compute_iou(y_true, y_pred, positive_class=1),
    }
