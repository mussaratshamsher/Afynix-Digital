from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    precision_recall_curve,
    f1_score,
    precision_score,
    recall_score,
    auc,
)


@dataclass(frozen=True)
class ThresholdResult:
    threshold: float
    metric_value: float


def compute_metrics(y_true, y_pred, y_proba, positive_label: int = 1) -> dict:
    metrics: dict = {}
    metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred).tolist()
    metrics["classification_report"] = classification_report(y_true, y_pred)

    # y_proba expected for positive class
    metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))

    precision, recall, thresholds = precision_recall_curve(y_true, y_proba, pos_label=positive_label)
    metrics["pr_auc"] = float(auc(recall, precision))

    metrics["precision_churn"] = float(precision_score(y_true, y_pred, pos_label=positive_label, zero_division=0))
    metrics["recall_churn"] = float(recall_score(y_true, y_pred, pos_label=positive_label, zero_division=0))
    metrics["f1"] = float(f1_score(y_true, y_pred, pos_label=positive_label, zero_division=0))
    return metrics


def tune_threshold(y_true, y_proba, metric: str = "f1") -> ThresholdResult:
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)

    # precision/recall have len = len(thresholds) + 1; align by using thresholds[...]
    # For evaluation, thresholds length is one less than precision/recall.
    best_t = 0.5
    best_v = -1.0
    for i, t in enumerate(thresholds):
        # Determine predictions at threshold t
        y_pred = (y_proba >= t).astype(int)
        if metric == "f1":
            v = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
        elif metric == "precision":
            v = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
        elif metric == "recall":
            v = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
        else:
            raise ValueError("metric must be one of: f1, precision, recall")

        if v > best_v:
            best_v = v
            best_t = float(t)

    return ThresholdResult(threshold=best_t, metric_value=float(best_v))

