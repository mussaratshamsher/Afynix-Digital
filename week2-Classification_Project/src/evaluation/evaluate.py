from __future__ import annotations

import os
import json

import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, RocCurveDisplay

from src.evaluation.metrics import compute_metrics, tune_threshold


def plot_curves(y_true, y_proba, out_dir: str, prefix: str = "model") -> None:
    os.makedirs(out_dir, exist_ok=True)

    # ROC
    RocCurveDisplay.from_predictions(y_true, y_proba)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{prefix}_roc_curve.png"))
    plt.close()

    # PR
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{prefix}_pr_curve.png"))
    plt.close()


def evaluate_model(model, X_test, y_test, out_dir: str, threshold_metric: str = "f1") -> dict:
    os.makedirs(out_dir, exist_ok=True)

    y_proba = model.predict_proba(X_test)[:, 1]
    threshold_result = tune_threshold(y_test, y_proba, metric=threshold_metric)

    y_pred = (y_proba >= threshold_result.threshold).astype(int)
    metrics = compute_metrics(y_test, y_pred, y_proba)
    metrics["threshold"] = threshold_result.threshold
    metrics["threshold_metric_value"] = threshold_result.metric_value

    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    plot_curves(y_test, y_proba, out_dir=out_dir, prefix="best")
    return metrics

