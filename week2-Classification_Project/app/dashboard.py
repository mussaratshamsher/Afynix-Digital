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

# Required columns expected by the churn model (must be defined before use)
REQUIRED_COLUMNS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges"
]

st.title("Customer Churn Prediction Dashboard")

# Configure backend API URL connection
st.sidebar.header("Backend")
default_api_url = os.getenv("API_URL", "https://mussarat123shamsher-classification-project.hf.space")
api_url = st.sidebar.text_input(
    "FastAPI Base URL",
    value=default_api_url,
    help="Base URL of the FastAPI backend. Leave default to use the HF Space backend."
)


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

# Show required columns in sidebar
with st.sidebar.expander("Required CSV Columns", expanded=False):
    st.markdown("Your CSV must include these columns:")
    st.code(", ".join(REQUIRED_COLUMNS))
    st.markdown("See `data/demo_upload.csv` for reference format.")

# Link to demo sample file for download
demo_csv_path = os.path.join(os.path.dirname(__file__), "..", "data", "demo_upload.csv")
if os.path.exists(demo_csv_path):
    with open(demo_csv_path, "rb") as f:
        st.sidebar.download_button(
            label="Download Sample CSV Template",
            data=f,
            file_name="sample_customers.csv",
            mime="text/csv",
            help="Download a sample CSV file with the correct columns"
        )

uploaded = st.file_uploader(
    "Upload a CSV with customer data (all 19 columns required)",
    type=["csv"],
    accept_multiple_files=False
)

if uploaded is not None:
    # Streamlit file objects sometimes need a more forgiving CSV parser.
    # Try a standard read first; if it fails, fall back to a delimiter/engine fallback.
    try:
        df = pd.read_csv(uploaded)
    except Exception:
        try:
            df = pd.read_csv(uploaded, sep=";", engine="python")
        except Exception as e:
            st.error(f"Failed to parse CSV file: {e}")
            st.stop()

    # Validate required columns
    uploaded_cols = set(df.columns)
    missing_cols = set(REQUIRED_COLUMNS) - uploaded_cols
    if missing_cols:
        st.error(f"Missing required columns: {sorted(missing_cols)}")
        st.info(f"Please upload a CSV with these columns: {REQUIRED_COLUMNS}")
        st.stop()

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
                response = requests.post(
                    f"{api_url}/predict_batch",
                    json={"data": records},
                    timeout=10
                )
                # On HF Spaces you often get a plain 500 page. Surface the body for debugging.
                if response.status_code != 200:
                    try:
                        err_payload = response.json()
                    except Exception:
                        err_payload = response.text
                    raise RuntimeError(f"Backend {response.status_code}: {err_payload}")

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

