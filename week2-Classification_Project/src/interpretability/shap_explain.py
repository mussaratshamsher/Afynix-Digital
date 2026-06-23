from __future__ import annotations

import json
import os

import numpy as np
import shap


def explain_with_shap(pipeline, X, out_dir: str, max_samples: int = 200) -> None:
    os.makedirs(out_dir, exist_ok=True)

    # Get fitted preprocessor and model
    preprocessor = pipeline.named_steps.get("preprocessor")
    model = pipeline.named_steps.get("model")

    # Transform X
    X_trans = preprocessor.transform(X)

    # Sample for speed
    if X_trans.shape[0] > max_samples:
        idx = np.random.RandomState(42).choice(X_trans.shape[0], max_samples, replace=False)
        X_trans_small = X_trans[idx]
        X_small = X.iloc[idx] if hasattr(X, "iloc") else X[idx]
    else:
        X_trans_small = X_trans
        X_small = X

    # Use TreeExplainer if underlying model is tree-based
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_trans_small)
    except Exception:
        # Fallback to KernelExplainer (slower; keep small)
        explainer = shap.KernelExplainer(model.predict_proba, shap.sample(X_trans_small, min(50, X_trans_small.shape[0])))
        shap_values = explainer.shap_values(X_trans_small)

    # Global bar plot
    shap.summary_plot(shap_values, X_trans_small, show=False)
    import matplotlib.pyplot as plt

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "shap_summary.png"))
    plt.close()

    # Save SHAP values stats
    stats = {
        "num_samples": int(X_trans_small.shape[0]),
        "num_features": int(X_trans_small.shape[1]),
    }
    with open(os.path.join(out_dir, "shap_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

