from __future__ import annotations

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.models.pipelines import make_preprocessor


def build_logistic_pipeline(X, imbalance: str, random_state: int = 42) -> Pipeline:
    # imbalance: 'class_weight' or 'none'
    class_weight = "balanced" if imbalance == "class_weight" else None

    preprocessor = make_preprocessor(X, model_type="logistic")
    model = LogisticRegression(
        class_weight=class_weight,
        max_iter=1000,
        solver="lbfgs",
        random_state=random_state,
    )

    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])

