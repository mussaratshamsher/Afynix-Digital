from __future__ import annotations

import os
import sys
import joblib
import pandas as pd
import streamlit as st

# Add the project root to sys.path so 'src' can be imported if running locally
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.features.build_features import build_features

st.set_page_config(page_title="Churn Dashboard", layout="wide")

# Required columns expected by the churn model
REQUIRED_COLUMNS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges"
]

st.title("Customer Churn Prediction Dashboard")

# Load local model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "churn_model.joblib")
MODEL_PATH = os.path.abspath(MODEL_PATH)

local_model_loaded = False
local_threshold = 0.5
local_model = None

if os.path.exists(MODEL_PATH):
    try:
        bundle = joblib.load(MODEL_PATH)
        local_model = bundle["model"]
        local_threshold = float(bundle.get("threshold", 0.5))
        local_model_loaded = True
    except Exception as e:
        st.error(f"Error loading model: {e}")
else:
    st.error("Model file not found. Please ensure churn_model.joblib is in the models folder.")

if not local_model_loaded:
    st.stop()

st.success(f"Model loaded successfully (threshold: {local_threshold:.4f})")

# Sidebar with template download
with st.sidebar:
    st.header("Sample File")
    demo_csv_path = os.path.join(os.path.dirname(__file__), "..", "data", "demo_upload.csv")
    if os.path.exists(demo_csv_path):
        with open(demo_csv_path, "rb") as f:
            st.download_button(
                label="Download Sample CSV",
                data=f,
                file_name="sample_customers.csv",
                mime="text/csv",
            )
    else:
        st.warning("Sample file not found")

    with st.expander("Required Columns"):
        st.code(", ".join(REQUIRED_COLUMNS))

# File upload
uploaded = st.file_uploader(
    "Upload a CSV with customer data",
    type=["csv"],
    accept_multiple_files=False
)

if uploaded is not None:
    # Parse CSV
    try:
        df = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Failed to parse CSV: {e}")
        st.stop()

    # Validate columns
    uploaded_cols = set(df.columns)
    missing_cols = set(REQUIRED_COLUMNS) - uploaded_cols
    if missing_cols:
        st.error(f"Missing columns: {sorted(missing_cols)}")
        st.info(f"Required: {REQUIRED_COLUMNS}")
        st.stop()

    # Remove Churn column if present (it's the target, not a feature)
    df_to_predict = df.copy()
    if "Churn" in df_to_predict.columns:
        df_to_predict = df_to_predict.drop(columns=["Churn"])

    # Make predictions
    with st.spinner("Making predictions..."):
        X_feat, _ = build_features(df_to_predict)
        proba = local_model.predict_proba(X_feat)[:, 1]

    def risk_tier(p: float) -> str:
        if p >= 0.66:
            return "High"
        if p >= 0.33:
            return "Medium"
        return "Low"

    # Build output
    out = df.copy()
    out["churn_probability"] = proba
    out["predicted_churn"] = (proba >= local_threshold).astype(int)
    out["risk_tier"] = [risk_tier(p) for p in proba]

    st.success("Predictions complete!")

    # Display results
    st.subheader("Top Risk Customers")
    st.dataframe(out.sort_values("churn_probability", ascending=False).head(25), use_container_width=True)

    st.subheader("Risk Distribution")
    risk_counts = out["risk_tier"].value_counts()
    st.bar_chart(risk_counts.rename(index={"Low": "Low Risk", "Medium": "Medium Risk", "High": "High Risk"}))