"""
Evaluation metrics for marine debris detection.

Placeholder module: extend with IoU, per-class precision/recall/F1,
confusion matrix utilities, and segmentation-specific metrics as the
project develops.
"""

import json
import os
from datetime import datetime

import numpy as np
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    confusion_matrix,
)


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
    """
    Return metrics for the binary debris (positive=1) vs non-debris (0) task.

    precision/recall/F1/IoU are reported for the debris (positive) class and use
    no true-negatives, so they are already robust to the class imbalance. MCC and
    balanced accuracy summarise both classes:

      - ``mcc`` (Matthews Correlation Coefficient, range [-1, 1]) is the
        recommended single imbalance-aware score -- it uses all four confusion
        cells and is high only when both classes are predicted well.
      - ``balanced_accuracy`` = mean of per-class recall; better than raw accuracy
        but, under extreme imbalance, specificity saturates so it stays optimistic.
      - ``accuracy`` is dominated by the non-debris majority and is *not* a
        reliable headline metric here -- kept only for reference.
    """
    return {
        "precision":         precision_score(y_true, y_pred, zero_division=0),
        "recall":            recall_score(y_true, y_pred, zero_division=0),
        "f1":                f1_score(y_true, y_pred, zero_division=0),
        "iou":               compute_iou(y_true, y_pred, positive_class=1),
        "mcc":               matthews_corrcoef(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "accuracy":          accuracy_score(y_true, y_pred),
    }


def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Binary confusion matrix with a fixed [0, 1] label order.

    Returns a 2x2 array laid out as::

        [[TN, FP],
         [FN, TP]]

    ``labels=[0, 1]`` is passed explicitly so the shape is stable even when a
    split contains only one class (e.g. a patch with no debris pixels).
    """
    return confusion_matrix(y_true, y_pred, labels=[0, 1])


def save_metrics(
    out_prefix: str,
    metrics: dict,
    report_text: str | None = None,
    extra: dict | None = None,
) -> None:
    """
    Persist evaluation results next to the model artifacts.

    Writes ``<out_prefix>_metrics.json`` (machine-readable, timestamped) and,
    when ``report_text`` is given, ``<out_prefix>_report.txt`` (the sklearn
    classification report for human inspection).

    Args:
        out_prefix: path prefix, e.g. "test/data/outputs/rf_baseline_test".
        metrics: dict of scalar metrics (from :func:`compute_binary_metrics`).
        report_text: optional classification-report string to save verbatim.
        extra: optional extra fields to record (e.g. sample counts, split name).
    """
    os.makedirs(os.path.dirname(out_prefix) or ".", exist_ok=True)

    payload = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "metrics": {k: float(v) for k, v in metrics.items()},
    }
    if extra:
        payload.update(extra)

    with open(f"{out_prefix}_metrics.json", "w") as f:
        json.dump(payload, f, indent=2)

    if report_text is not None:
        with open(f"{out_prefix}_report.txt", "w") as f:
            f.write(report_text)
