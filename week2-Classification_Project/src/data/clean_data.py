from __future__ import annotations

import pandas as pd


def normalize_target(df: pd.DataFrame, target_col: str = "Churn", positive_label: str = "Yes") -> pd.DataFrame:
    df = df.copy()
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found. Columns: {list(df.columns)}")

    # Map common variants to 0/1
    y_raw = df[target_col]
    if y_raw.dtype == bool:
        df[target_col] = y_raw.astype(int)
        return df

    if y_raw.dtype.kind in {"i", "u", "f"}:
        # assume already 0/1
        df[target_col] = (y_raw > 0).astype(int)
        return df

    y_str = y_raw.astype(str).str.strip().str.lower()
    df[target_col] = (y_str == positive_label.lower()).astype(int)
    return df


def clean_dataframe(df: pd.DataFrame, target_col: str = "Churn") -> pd.DataFrame:
    df = df.copy()

    # Drop obvious identifier columns if present
    for col in ["customerID", "CustomerId", "id", "ID"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    # Coerce any numeric-like columns stored as object into numeric
    for col in df.columns:
        if col == target_col:
            continue
        if df[col].dtype == "object":
            # if many values look numeric, coerce
            coerced = pd.to_numeric(df[col].replace({"": pd.NA, " ": pd.NA}), errors="coerce")
            # If coercion introduces NaNs but also produces many non-NaN numeric values, keep it.
            non_nan_before = df[col].notna().sum()
            non_nan_after = coerced.notna().sum()
            if non_nan_after >= max(10, int(0.6 * non_nan_before)):
                df[col] = coerced

    # Basic missing handling: median for numeric, most_frequent for categorical
    # (Imputation later also exists inside pipelines; this is just lightweight robustness)
    return df

