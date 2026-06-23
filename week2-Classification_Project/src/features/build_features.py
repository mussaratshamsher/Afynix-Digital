from __future__ import annotations

import numpy as np
import pandas as pd


def build_features(df: pd.DataFrame, target_col: str = "Churn") -> tuple[pd.DataFrame, pd.Series | None]:
    df = df.copy()
    if target_col in df.columns:
        y = df[target_col]
        X = df.drop(columns=[target_col])
    else:
        y = None
        X = df

    # Telecom-inspired feature engineering (safe if cols absent)
    if "tenure" in X.columns and "TotalCharges" in X.columns:
        # average monthly spend; avoid div-by-zero
        tenure = pd.to_numeric(X["tenure"], errors="coerce")
        total = pd.to_numeric(X["TotalCharges"], errors="coerce")
        X["avg_monthly_spend"] = np.where(tenure > 0, total / tenure, np.nan)

    if "Contract" in X.columns:
        # risk flag: month-to-month likely higher churn
        X["contract_risk_flag"] = X["Contract"].astype(str).str.lower().str.contains("month-to-month").astype(int)

    # Count-of-services heuristic
    service_cols = [
        "PhoneService",
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ]
    present = [c for c in service_cols if c in X.columns]
    if present:
        # treat presence of non-empty/Non-No as subscribed
        def subscribed(series: pd.Series) -> pd.Series:
            s = series.astype(str).str.strip().str.lower()
            return (~s.isin(["", "none", "no"]))

        X["services_subscribed_count"] = sum(subscribed(X[c]) for c in present)

    return X, y

