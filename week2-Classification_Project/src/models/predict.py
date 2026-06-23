from __future__ import annotations

import numpy as np


def predict_proba(model, X) -> np.ndarray:
    # model should be sklearn Pipeline with predict_proba
    return model.predict_proba(X)[:, 1]


def predict(model, X, threshold: float = 0.5) -> np.ndarray:
    proba = predict_proba(model, X)
    return (proba >= threshold).astype(int)

