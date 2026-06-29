from __future__ import annotations

import os
import sys
import joblib
from typing import List, Dict, Any

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import pandas as pd

# Add the project root to sys.path so 'src' can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.features.build_features import build_features

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "churn_model.joblib")
MODEL_PATH = os.path.abspath(MODEL_PATH)

bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
threshold = float(bundle.get("threshold", 0.5))

class CustomerInput(BaseModel):
    data: dict

class BatchCustomerInput(BaseModel):
    data: List[dict]

app = FastAPI(title="Customer Churn Prediction")

# Enable CORS for cross-origin Streamlit integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def risk_tier(prob: float) -> str:
    # Simple tiering: Low < 0.33, Medium 0.33-0.66, High >= 0.66
    if prob >= 0.66:
        return "High"
    if prob >= 0.33:
        return "Medium"
    return "Low"

@app.get("/")
def home():
    return {
        "status": "healthy",
        "model_loaded": True,
        "threshold": threshold,
        "message": "Customer Churn Prediction API is running."
    }

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

@app.post("/predict_batch")
def predict_batch(payload: BatchCustomerInput):
    import pandas as pd

    X = pd.DataFrame(payload.data)
    X_feat, _ = build_features(X)
    res = model.predict_proba(X_feat)
    probas = res[:, 1].tolist()
    preds = [int(p >= threshold) for p in probas]
    tiers = [risk_tier(p) for p in probas]

    return {
        "churn_probabilities": probas,
        "threshold": threshold,
        "predicted_churns": preds,
        "risk_tiers": tiers,
    }

# Required columns for CSV upload
REQUIRED_COLUMNS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges"
]

@app.post("/predict_csv")
async def predict_csv(file: UploadFile = File(...)):
    """
    Upload a CSV file with customer data for batch predictions.
    The CSV must contain the required columns.
    """
    # Read CSV file
    try:
        contents = await file.read()
        from io import StringIO
        df = pd.read_csv(StringIO(contents.decode("utf-8")))
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"error": f"Failed to parse CSV: {str(e)}"}
        )

    # Validate columns
    missing_cols = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing_cols:
        return JSONResponse(
            status_code=400,
            content={
                "error": f"Missing required columns: {sorted(missing_cols)}",
                "required": REQUIRED_COLUMNS
            }
        )

    # Store customer IDs if present (not a feature)
    customer_ids = df["customerID"].tolist() if "customerID" in df.columns else None
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    # Drop Churn column if present (it's the target, not a feature)
    if "Churn" in df.columns:
        df = df.drop(columns=["Churn"])

    # Make predictions
    X_feat, _ = build_features(df)
    res = model.predict_proba(X_feat)
    probas = res[:, 1].tolist()
    preds = [int(p >= threshold) for p in probas]
    tiers = [risk_tier(p) for p in probas]

    # Build response
    result = {
        "predictions": [
            {
                "churn_probability": proba,
                "predicted_churn": pred,
                "risk_tier": tier,
            }
            for proba, pred, tier in zip(probas, preds, tiers)
        ],
        "threshold": threshold,
        "total_records": len(probas),
    }

    # Add customer IDs if available
    if customer_ids:
        for i, r in enumerate(result["predictions"]):
            r["customerID"] = customer_ids[i]

    return result

