from __future__ import annotations

import os
import joblib

import pandas as pd
import streamlit as st


MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "churn_model.joblib")
MODEL_PATH = os.path.abspath(MODEL_PATH)

bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
threshold = float(bundle.get("threshold", 0.5))

st.set_page_config(page_title="Churn Dashboard", layout="wide")

st.title("Customer Churn Risk Dashboard")
st.write(f"Using decision threshold: **{threshold:.4f}**")

uploaded = st.file_uploader("Upload a CSV of customer rows (must match training columns)", type=["csv"], accept_multiple_files=False)

import sys
import os

# Add the project root to sys.path so 'src' can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.features.build_features import build_features

if uploaded is not None:
    df = pd.read_csv(uploaded)

    if "Churn" in df.columns:
        df = df.drop(columns=["Churn"])

    X_feat, _ = build_features(df)
    proba = model.predict_proba(X_feat)[:, 1]
    out = df.copy()
    out["churn_probability"] = proba
    out["predicted_churn"] = (out["churn_probability"] >= threshold).astype(int)

    st.subheader("Top Risk Customers")
    st.dataframe(out.sort_values("churn_probability", ascending=False).head(25), use_container_width=True)

    st.subheader("Risk distribution")
    st.bar_chart(out["predicted_churn"].value_counts().rename(index={0: "No churn", 1: "Churn"}))

