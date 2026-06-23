from __future__ import annotations

import os
import sys
import joblib
import pandas as pd
import streamlit as st
import requests

# Add the project root to sys.path so 'src' can be imported if running locally
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.features.build_features import build_features

st.set_page_config(page_title="Churn Dashboard", layout="wide")

st.title("Customer Churn Risk Dashboard")

# Configure backend API URL connection
st.sidebar.header("Backend API Connection")
default_api_url = os.getenv("API_URL", "https://mussarat123shamsher-classification-project.hf.space")
api_url = st.sidebar.text_input("FastAPI Base URL", value=default_api_url)

# Test API connection
api_connected = False
try:
    if api_url:
        response = requests.get(f"{api_url}/", timeout=3)
        if response.status_code == 200:
            api_connected = True
            st.sidebar.success("Connected to API!")
        else:
            st.sidebar.warning(f"API returned status {response.status_code}")
except Exception as e:
    st.sidebar.warning("Could not reach API. Will use local fallback if enabled.")

# Determine if we should use API or Local Fallback
use_local = False
if not api_connected:
    use_local = st.sidebar.checkbox("Fallback to Local Model (if available)", value=True)
else:
    use_local = st.sidebar.checkbox("Force Local Model", value=False)

# Load local model metadata / threshold if we are in local mode or to show threshold
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
        st.sidebar.error(f"Error loading local model: {e}")

# Display Decision Threshold
if use_local:
    st.info(f"Running mode: **Local Model** (Threshold: **{local_threshold:.4f}**)")
else:
    # Try to fetch current configuration from API
    try:
        api_info = requests.get(f"{api_url}/", timeout=3).json()
        api_threshold = api_info.get("threshold", 0.5)
        st.info(f"Running mode: **Remote API** (Threshold: **{api_threshold:.4f}**)")
    except Exception:
        st.info("Running mode: **Remote API** (Threshold: Unknown)")

uploaded = st.file_uploader(
    "Upload a CSV of customer rows (must match training columns)",
    type=["csv"],
    accept_multiple_files=False
)

if uploaded is not None:
    df = pd.read_csv(uploaded)

    # Clean Churn columns if they exist in the raw test CSV
    df_to_predict = df.copy()
    if "Churn" in df_to_predict.columns:
        df_to_predict = df_to_predict.drop(columns=["Churn"])

    # Perform predictions
    if not use_local and api_connected:
        st.write("Sending request to FastAPI backend...")
        records = df_to_predict.to_dict(orient="records")
        try:
            with st.spinner("Requesting predictions from API..."):
                response = requests.post(f"{api_url}/predict_batch", json={"data": records}, timeout=10)
                response.raise_for_status()
                res_data = response.json()
            
            out = df.copy()
            out["churn_probability"] = res_data["churn_probabilities"]
            out["predicted_churn"] = res_data["predicted_churns"]
            out["risk_tier"] = res_data["risk_tiers"]
            st.success("Successfully fetched predictions from FastAPI!")
        except Exception as e:
            st.error(f"Failed to fetch predictions from API: {e}")
            st.stop()
    else:
        # Fallback to local prediction
        if not local_model_loaded:
            st.error("No local model found or loaded. Please start the backend API or place the model file in the models directory.")
            st.stop()
        
        st.write("Using local model fallback...")
        X_feat, _ = build_features(df_to_predict)
        proba = local_model.predict_proba(X_feat)[:, 1]
        
        def risk_tier(p: float) -> str:
            if p >= 0.66:
                return "High"
            if p >= 0.33:
                return "Medium"
            return "Low"

        out = df.copy()
        out["churn_probability"] = proba
        out["predicted_churn"] = (out["churn_probability"] >= local_threshold).astype(int)
        out["risk_tier"] = [risk_tier(p) for p in proba]
        st.success("Successfully generated local predictions!")

    # Display results
    st.subheader("Top Risk Customers")
    st.dataframe(out.sort_values("churn_probability", ascending=False).head(25), use_container_width=True)

    st.subheader("Risk Distribution")
    st.bar_chart(out["predicted_churn"].value_counts().rename(index={0: "No churn", 1: "Churn"}))

