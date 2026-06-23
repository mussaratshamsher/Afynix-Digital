from __future__ import annotations

import os
import joblib

from fastapi import FastAPI
from pydantic import BaseModel


MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "churn_model.joblib")
MODEL_PATH = os.path.abspath(MODEL_PATH)

bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
threshold = float(bundle.get("threshold", 0.5))


class CustomerInput(BaseModel):
    # Accept arbitrary key/values for flexibility
    data: dict


app = FastAPI(title="Customer Churn Prediction")


def risk_tier(prob: float) -> str:
    # Simple tiering: Low < 0.33, Medium 0.33-0.66, High >= 0.66
    if prob >= 0.66:
        return "High"
    if prob >= 0.33:
        return "Medium"
    return "Low"


import sys
import os

# Add the project root to sys.path so 'src' can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.features.build_features import build_features


@app.post("/predict")
def predict(payload: CustomerInput):
    # Ensure single row dataframe
    import pandas as pd

    X = pd.DataFrame([payload.data])
    X_feat, _ = build_features(X)
    res = model.predict_proba(X_feat)
    proba = float(res[0, 1])
    pred = int(proba >= threshold)

    return {
        "churn_probability": proba,
        "threshold": threshold,
        "predicted_churn": pred,
        "risk_tier": risk_tier(proba),
    }

