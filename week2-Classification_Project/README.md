# Week 2 — Customer Churn Prediction (Classification)

This project implements an end-to-end customer churn classification pipeline:
- Data loading + cleaning
- Encoding + feature engineering
- Imbalance handling (class_weight / SMOTE)
- Model training + evaluation (precision/recall/F1, ROC-AUC, PR-AUC)
- Threshold tuning
- SHAP explanations
- FastAPI inference endpoint
- Streamlit dashboard

## Expected dataset
Place a CSV at:
- `data/raw/churn.csv`

The pipeline expects a target column named `Churn` with values like `Yes/No` (or `1/0`).

## Setup
Create a virtual environment and install dependencies:

```bash
python -m venv .venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

## Train
Run training:

```bash
python week2-Classification_Project/src/train.py \
  --data_path week2-Classification_Project/data/raw/churn.csv \
  --output_dir week2-Classification_Project
```

Artifacts will be written to:
- `week2-Classification_Project/models/`
- `week2-Classification_Project/reports/`
- `week2-Classification_Project/outputs/`

## Start API
```bash
uvicorn week2-Classification_Project.app.api:app --reload --port 8000
```

Then call:
- `POST http://127.0.0.1:8000/predict`

## Start Dashboard
```bash
streamlit run week2-Classification_Project/app/dashboard.py
```

## Optional: LLM insights (Groq)
If `GROQ_API_KEY` is set, `src/llm_insights.py` will generate a narrative summary using Groq.

## Notes
- Resampling (SMOTE) is applied only to training folds (no leakage).
- The default probability threshold is tuned using PR curves.

