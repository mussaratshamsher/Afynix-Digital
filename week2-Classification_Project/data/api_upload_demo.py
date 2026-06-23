"""API vs Streamlit upload demo generator.

Creates a small CSV file in this folder that can be uploaded in Streamlit,
and also converted to a JSON payload to send to the FastAPI backend.

Usage:
  python data/api_upload_demo.py

Outputs:
  - demo_upload.csv
  - demo_payload.json

Notes:
- The FastAPI endpoint expects the same feature columns that the model was trained on.
- This script cannot guarantee column compatibility with the trained pipeline,
  but it provides a clear template for comparing:
    1) Streamlit -> API POST /predict_batch
    2) Local model fallback inside Streamlit
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent

OUT_CSV = HERE / "demo_upload.csv"
OUT_JSON = HERE / "demo_payload.json"


def main() -> None:
    # Minimal example rows. Replace/extend columns to match your training columns.
    # This is intentionally generic so you can edit values to fit your dataset.
    df = pd.DataFrame(
        [
            {
                "CustomerID": "C001",
                "Age": 35,
                "TenureMonths": 12,
                "MonthlyCharges": 70.5,
                "TotalCharges": 840.0,
                "Gender": "Male",
                "Contract": "Month-to-month",
                "InternetService": "DSL",
                "PaymentMethod": "Electronic check",
            },
            {
                "CustomerID": "C002",
                "Age": 52,
                "TenureMonths": 40,
                "MonthlyCharges": 110.2,
                "TotalCharges": 4408.0,
                "Gender": "Female",
                "Contract": "Two year",
                "InternetService": "Fiber optic",
                "PaymentMethod": "Credit card",
            },
        ]
    )

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)

    payload = {"data": df.to_dict(orient="records")}
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Wrote CSV: {OUT_CSV}")
    print(f"Wrote JSON payload: {OUT_JSON}")
    print("\nUse the CSV in Streamlit file uploader.")
    print("Then, send the JSON to the backend endpoint: POST /predict_batch")


if __name__ == "__main__":
    main()

