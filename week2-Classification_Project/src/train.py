from __future__ import annotations

import argparse
import os
import joblib

import numpy as np
from sklearn.model_selection import train_test_split

from src.data.load_data import load_csv
from src.data.clean_data import clean_dataframe, normalize_target
from src.features.build_features import build_features
from src.models.train_logistic import build_logistic_pipeline
from src.models.train_tree_models import build_random_forest_pipeline
from src.evaluation.evaluate import evaluate_model
from src.interpretability.shap_explain import explain_with_shap
from src.utils.config import TrainConfig


def imbalance_report(y):
    y = np.asarray(y)
    pos = int(y.sum())
    neg = int((1 - y).sum())
    total = pos + neg
    return {
        "positive_count": pos,
        "negative_count": neg,
        "positive_rate": float(pos / total) if total else 0.0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    cfg = TrainConfig()

    data_path = args.data_path
    output_dir = args.output_dir

    raw = load_csv(data_path)
    raw = normalize_target(raw, target_col=cfg.target_col, positive_label=cfg.positive_label)
    raw = clean_dataframe(raw, target_col=cfg.target_col)

    X, y = build_features(raw, target_col=cfg.target_col)

    # Split first (no leakage)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=cfg.test_size,
        stratify=y,
        random_state=cfg.random_state,
    )

    os.makedirs(os.path.join(output_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "reports"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "outputs"), exist_ok=True)

    # Save imbalance report
    with open(os.path.join(output_dir, "reports", "imbalance.json"), "w", encoding="utf-8") as f:
        import json

        json.dump(imbalance_report(y_train), f, indent=2)

    # Logistic baseline with class_weight
    log_model = build_logistic_pipeline(X_train, imbalance="class_weight", random_state=cfg.random_state)
    log_model.fit(X_train, y_train)

    log_metrics = evaluate_model(
        log_model,
        X_test,
        y_test,
        out_dir=os.path.join(output_dir, "reports", "logistic"),
        threshold_metric=cfg.threshold_metric,
    )

    # Random forest with SMOTE (inside pipeline)
    rf_model = build_random_forest_pipeline(X_train, imbalance="smote", random_state=cfg.random_state)
    rf_model.fit(X_train, y_train)

    rf_metrics = evaluate_model(
        rf_model,
        X_test,
        y_test,
        out_dir=os.path.join(output_dir, "reports", "random_forest_smote"),
        threshold_metric=cfg.threshold_metric,
    )

    # Pick best by threshold_metric_value (primary) then PR-AUC
    def score(m):
        return float(m.get("threshold_metric_value", -1)) + 0.000001 * float(m.get("pr_auc", 0))

    best_model = rf_model if score(rf_metrics) >= score(log_metrics) else log_model
    best_metrics = rf_metrics if best_model is rf_model else log_metrics

    joblib.dump(
        {
            "model": best_model,
            "threshold": best_metrics["threshold"],
            "cfg": cfg.__dict__,
        },
        os.path.join(output_dir, "models", "churn_model.joblib"),
    )

    # SHAP explanation (on best tree model ideally)
    try:
        explain_with_shap(best_model, X_test, out_dir=os.path.join(output_dir, "reports", "shap"))
    except Exception as e:
        print(f"SHAP failed: {e}")

    # Write summary comparison
    with open(os.path.join(output_dir, "reports", "model_comparison.md"), "w", encoding="utf-8") as f:
        f.write("# Model Comparison\n\n")
        f.write("## Logistic Regression (class_weight)\n")
        f.write(f"- Threshold: {log_metrics['threshold']}\n")
        f.write(f"- PR-AUC: {log_metrics['pr_auc']}\n")
        f.write(f"- Precision(churn): {log_metrics['precision_churn']}\n")
        f.write(f"- Recall(churn): {log_metrics['recall_churn']}\n")
        f.write(f"- F1: {log_metrics['f1']}\n\n")

        f.write("## Random Forest (SMOTE)\n")
        f.write(f"- Threshold: {rf_metrics['threshold']}\n")
        f.write(f"- PR-AUC: {rf_metrics['pr_auc']}\n")
        f.write(f"- Precision(churn): {rf_metrics['precision_churn']}\n")
        f.write(f"- Recall(churn): {rf_metrics['recall_churn']}\n")
        f.write(f"- F1: {rf_metrics['f1']}\n\n")

        f.write("## Selected Model\n")
        f.write(f"- Selected based on: {cfg.threshold_metric}\n")
        f.write(f"- Threshold: {best_metrics['threshold']}\n")
        f.write(f"- PR-AUC: {best_metrics['pr_auc']}\n")

    print("Training complete. Best model saved to models/churn_model.joblib")


if __name__ == "__main__":
    main()

