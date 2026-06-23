# Customer Churn Prediction — Project Architecture & Planning Document

**Project Codename:** *Save the Business*
**Track:** Week 2 — Classification Project
**Author:** Solution Architecture Team
**Status:** Planning / Pre-Development

---

## 1. Executive Summary

Customer churn is one of the most expensive problems a subscription or service-based business faces — it is far cheaper to retain an existing customer than to acquire a new one. This project builds a **binary classification system** that predicts whether a customer is likely to churn, so the business can intervene early (discounts, support outreach, contract renegotiation) before the customer leaves.

This document defines the **end-to-end architecture, data strategy, modeling approach, evaluation methodology, and delivery plan** for the project. It is written to be implementation-ready — an engineer should be able to pick this up and start building directly from this plan.

### 1.1 Problem Statement
> Given historical customer data (demographics, account info, usage/service patterns, billing details), predict the probability that a customer will churn, and provide the business with **interpretable, actionable risk segments**, not just a yes/no label.

### 1.2 Business Success Criteria
| Goal | Metric | Target |
|---|---|---|
| Catch most at-risk customers | Recall (churn class) | ≥ 0.75 |
| Avoid wasting retention budget | Precision (churn class) | ≥ 0.55 |
| Overall ranking quality | ROC-AUC | ≥ 0.80 |
| Business cost reduction | Net savings (TP value − FP cost) | Positive & maximized |

> ⚠️ Accuracy is **explicitly excluded** as a primary success metric — see Section 7.

---

## 2. System Architecture

### 2.1 High-Level Pipeline

```
┌───────────────┐     ┌──────────────┐     ┌────────────────┐     ┌───────────────┐
│  Raw Data     │ --> │  Data Prep   │ --> │  Feature Eng.  │ --> │  Imbalance    │
│  (CSV / DB)   │     │  & Cleaning  │     │  & Encoding    │     │  Handling     │
└───────────────┘     └──────────────┘     └────────────────┘     └───────────────┘
                                                                            │
                                                                            ▼
┌───────────────┐     ┌──────────────┐     ┌────────────────┐     ┌───────────────┐
│  Deployment   │ <-- │  Model       │ <-- │  Model         │ <-- │  Train/Test   │
│  (API / App)  │     │  Evaluation  │     │  Training      │     │  Split        │
└───────────────┘     └──────────────┘     └────────────────┘     └───────────────┘
```

### 2.2 Logical Components

| Layer | Responsibility | Key Tools |
|---|---|---|
| Data Layer | Ingestion, storage, versioning of raw/processed data | pandas, CSV/Parquet, (optional) PostgreSQL |
| Processing Layer | Cleaning, encoding, scaling, feature engineering | pandas, scikit-learn `Pipeline` |
| Imbalance Layer | Resampling or weighted loss | imbalanced-learn (SMOTE), `class_weight` |
| Modeling Layer | Train, tune, compare models | scikit-learn, XGBoost / LightGBM |
| Evaluation Layer | Metrics, plots, business cost analysis | scikit-learn.metrics, matplotlib/seaborn |
| Serving Layer | Expose predictions | FastAPI / Flask, joblib |
| Presentation Layer | Visualize churn risk for business users | Streamlit / simple dashboard |
| Experiment Tracking | Compare model runs | MLflow (optional, recommended) |

### 2.3 Recommended Tech Stack

```
Language:        Python 3.10+
Data Handling:    pandas, numpy
Visualization:    matplotlib, seaborn, plotly (optional)
Imbalance:        imbalanced-learn (SMOTE, RandomOverSampler)
ML Models:        scikit-learn, xgboost, lightgbm
Tuning:           GridSearchCV / RandomizedSearchCV / Optuna
Interpretability: shap
Serving:          FastAPI, uvicorn, joblib/pickle
Dashboard:        Streamlit
Tracking:         MLflow (optional)
Env Management:   venv 
```

---

## 3. Project Folder Structure

```
customer-churn-app/
│
├── data/
│   ├── raw/                  # original, untouched dataset
│   ├── interim/              # intermediate cleaned data
│   └── processed/            # model-ready data
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_imbalance_experiments.ipynb
│   ├── 04_modeling_baseline.ipynb
│   └── 05_modeling_tree_based.ipynb
│
├── src/
│   ├── data/
│   │   ├── load_data.py
│   │   └── clean_data.py
│   ├── features/
│   │   └── build_features.py
│   ├── imbalance/
│   │   └── resampling.py
│   ├── models/
│   │   ├── train_logistic.py
│   │   ├── train_tree_models.py
│   │   └── predict.py
│   ├── evaluation/
│   │   └── metrics.py
│   └── utils/
│       └── config.py
│
├── models/                   # serialized trained models (.pkl/.joblib)
├── reports/
│   ├── figures/
│   └── model_comparison.md
│
├── app/
│   ├── api.py                # FastAPI inference endpoint
│   └── dashboard.py           # Streamlit dashboard
│
├── tests/
│   └── test_pipeline.py
│
├── requirements.txt
├── config.yaml
└── README.md
```

---

## 4. Data Understanding & Exploratory Data Analysis (EDA)

### 4.1 Expected Dataset Shape
Typical churn datasets include:
- **Demographics:** gender, senior citizen, partner, dependents
- **Account info:** tenure, contract type, payment method, billing type
- **Service usage:** internet service, phone service, streaming/add-ons
- **Target:** `Churn` (Yes/No)

### 4.2 Step 1 — Class Imbalance Check (Mandatory First Step)

```python
df['Churn'].value_counts()
df['Churn'].value_counts(normalize=True) * 100
sns.countplot(x='Churn', data=df)
```

This is the **first diagnostic** before anything else. Churn datasets are typically imbalanced (commonly ~70/30 or more skewed). The result of this step determines:
- Whether resampling is required
- Whether `class_weight='balanced'` should be used
- Which evaluation metrics matter most (Section 7)

### 4.3 Additional EDA Tasks
- Missing value audit (`df.isnull().sum()`)
- Data type audit (`df.dtypes`) — flag numeric columns stored as strings (e.g., `TotalCharges`)
- Univariate analysis of categorical features vs churn rate
- Correlation heatmap for numeric features
- Outlier detection on tenure / charges
- Churn rate by segment (contract type, tenure bucket, payment method) — these usually reveal the strongest signals

---

## 5. Data Preprocessing & Feature Engineering

### 5.1 Cleaning
- Convert incorrectly typed numeric columns (`TotalCharges` often loads as object due to blank strings)
- Drop or impute missing values (prefer imputation over dropping rows where possible)
- Drop identifier columns not useful for modeling (e.g., `customerID`)

### 5.2 Categorical Encoding Strategy

| Column Type | Example | Encoding Method |
|---|---|---|
| Binary categorical | `Partner`, `Dependents`, `PhoneService` | Label Encoding (0/1) |
| Nominal, low cardinality | `InternetService`, `PaymentMethod`, `Contract` | One-Hot Encoding |
| Ordinal | (if any tiered service levels) | Ordinal Encoding |
| High cardinality (if present) | `zip_code`, `customer_segment_id` | Target/Frequency Encoding |

> Rule of thumb: never feed raw strings into scikit-learn models — always encode. Use `pd.get_dummies()` or `OneHotEncoder` inside a scikit-learn `Pipeline`/`ColumnTransformer` so the encoding is reproducible at inference time.

### 5.3 Feature Engineering Ideas
- **Tenure buckets**: 0–12, 13–24, 25–48, 49+ months
- **Total services subscribed** (count of add-on services)
- **Average monthly spend** = `TotalCharges / tenure`
- **Contract risk flag**: month-to-month vs. long-term contracts
- **Payment method risk flag**: electronic check historically correlates with higher churn in many telecom datasets

### 5.4 Scaling
- Tree-based models: scaling not required
- Logistic Regression: **required** — use `StandardScaler` or `MinMaxScaler`

### 5.5 Train/Test Split
```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
```
> `stratify=y` is mandatory here — it preserves the churn/no-churn ratio in both splits, which matters a lot once imbalance is confirmed in Section 4.2.

---

## 6. Handling Class Imbalance

Once Section 4.2 confirms imbalance, choose a strategy based on severity:

| Imbalance Severity | Recommended Approach |
|---|---|
| Mild (60/40 – 70/30) | `class_weight='balanced'` is usually sufficient |
| Moderate (75/25 – 85/15) | SMOTE oversampling on training data only, or combine with `class_weight` |
| Severe (>85/15) | SMOTE + undersampling combo (e.g., `SMOTEENN`), threshold tuning, anomaly-style framing |

### 6.1 Techniques to Implement

```python
# Option A: class_weight (simplest, no data leakage risk)
LogisticRegression(class_weight='balanced')
RandomForestClassifier(class_weight='balanced')

# Option B: SMOTE oversampling — apply ONLY on training fold, never on test data
from imblearn.over_sampling import SMOTE
X_train_res, y_train_res = SMOTE(random_state=42).fit_resample(X_train, y_train)
```

### 6.2 Critical Rule
> **Resampling must happen inside the cross-validation fold, after the train/test split — never before.** Resampling before splitting leaks information from the test set into training and produces falsely optimistic results. Use `imblearn.pipeline.Pipeline` to wrap resampling + model together safely.

---

## 7. Evaluation Strategy — "Don't Rely Only on Accuracy"

### 7.1 Why Accuracy is Misleading
If 80% of customers don't churn, a model that predicts "no churn" for everyone scores **80% accuracy** while being completely useless. Accuracy on imbalanced classes hides poor performance on the minority (churn) class — which is the class the business actually cares about.

### 7.2 Required Metrics

| Metric | What It Tells the Business |
|---|---|
| **Confusion Matrix** | Raw counts of correct/incorrect predictions per class |
| **Precision (churn class)** | Of customers flagged as "will churn," how many actually do? (controls wasted retention spend) |
| **Recall (churn class)** | Of customers who actually churn, how many did we catch? (controls missed revenue loss) |
| **F1-score** | Harmonic balance of precision & recall |
| **ROC-AUC** | Overall ability to rank churners above non-churners across thresholds |
| **PR-AUC (Precision-Recall AUC)** | More informative than ROC-AUC under heavy imbalance |
| **Classification report** | Per-class breakdown, single command via `sklearn.metrics.classification_report` |

```python
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, precision_recall_curve, f1_score
)
print(classification_report(y_test, y_pred))
roc_auc_score(y_test, y_pred_proba)
```

### 7.3 Business Cost Framing (Recommended Add-On)
Translate the confusion matrix into dollars:

```
Net Value = (TP × avg_customer_value_saved) 
          − (FP × cost_of_retention_offer) 
          − (FN × avg_customer_lifetime_value_lost)
```

This reframes model selection as a **business optimization problem**, not just a statistics exercise, and is the most persuasive artifact to present to stakeholders.

### 7.4 Threshold Tuning
Default 0.5 probability threshold is rarely optimal under imbalance. Use the precision-recall curve to pick a threshold that matches business risk appetite (e.g., prioritize recall if churn cost > false-positive cost).

---

## 8. Model Strategy — Simple to Complex

### Phase 1 — Baseline: Logistic Regression
- Purpose: establish an interpretable, fast benchmark
- Use `class_weight='balanced'`, scaled features, and L2 regularization
- Inspect coefficients to validate "sane" business signal (e.g., short tenure → positive churn coefficient)

```python
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(class_weight='balanced', max_iter=1000)
model.fit(X_train_scaled, y_train)
```

### Phase 2 — Tree-Based Models
Progress in order of complexity:

1. **Decision Tree** — quick interpretability check, prone to overfitting
2. **Random Forest** — bagged ensemble, handles non-linear interactions, robust baseline upgrade
3. **Gradient Boosting (XGBoost / LightGBM)** — typically best raw performance on tabular churn data; supports `scale_pos_weight` for imbalance natively

```python
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(class_weight='balanced', n_estimators=200, random_state=42)

import xgboost as xgb
xgb_model = xgb.XGBClassifier(
    scale_pos_weight=(neg_count / pos_count),
    eval_metric='aucpr',
    use_label_encoder=False
)
```

### Phase 3 — Tuning & Selection
- `RandomizedSearchCV` or `GridSearchCV` over key hyperparameters (tree depth, learning rate, n_estimators, min_samples_leaf)
- Always tune using **cross-validation with stratified folds** (`StratifiedKFold`)
- Select the final model using the metric prioritized in Section 1.2 (recall/F1/PR-AUC) — not accuracy

### 8.1 Model Comparison Template

| Model | Precision (Churn) | Recall (Churn) | F1 | ROC-AUC | PR-AUC | Notes |
|---|---|---|---|---|---|---|
| Logistic Regression (baseline) | | | | | | Interpretable benchmark |
| Decision Tree | | | | | | Sanity check |
| Random Forest | | | | | | |
| XGBoost / LightGBM | | | | | | Likely best performer |

*(Table to be filled in during experimentation — store final results in `reports/model_comparison.md`)*

---

## 9. Interpretability & Stakeholder Communication

- Use **SHAP values** on the final tree-based model to explain individual predictions ("why was this customer flagged?")
- Generate a **global feature importance chart** for the business team
- Translate top features into plain-language churn drivers (e.g., "month-to-month contracts and low tenure are the strongest churn signals")

---

## 10. Deployment Plan

| Step | Description |
|---|---|
| 1. Serialize model | `joblib.dump(model, 'models/churn_model.joblib')` along with the fitted encoder/scaler |
| 2. Build inference API | FastAPI endpoint `/predict` accepting customer JSON, returning churn probability + risk tier |
| 3. Build business dashboard | Streamlit app showing at-risk customer list, filterable by segment, with SHAP explanation panel |
| 4. (Optional) Containerize | Dockerfile for consistent deployment |
| 5. Monitoring | Track prediction drift and periodically re-validate against new labeled data |

---

## 11. Implementation Roadmap

| Day | Milestone |
|---|---|
| Day 1 | Data loading, class imbalance check, full EDA |
| Day 2 | Data cleaning, encoding, feature engineering pipeline |
| Day 3 | Imbalance handling experiments (class_weight vs SMOTE) |
| Day 4 | Logistic Regression baseline + evaluation |
| Day 5 | Decision Tree + Random Forest training & comparison |
| Day 6 | XGBoost/LightGBM training, hyperparameter tuning |
| Day 7 | Final evaluation, SHAP interpretability, business cost analysis |
| Day 8 | Build FastAPI + Streamlit dashboard |
| Day 9 | Testing, documentation, README, packaging |
| Day 10 | Final review & presentation of results |

---

## 12. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Data leakage from resampling before split | Always split first; resample only inside training fold using `imblearn.Pipeline` |
| Overfitting on tree models | Cross-validation, depth limits, regularization, early stopping (boosting) |
| Choosing model based on accuracy alone | Enforce F1/Recall/PR-AUC as primary selection metric (Section 7) |
| Encoding mismatch between training and inference | Persist the fitted `ColumnTransformer`/encoder alongside the model |
| Stakeholders distrust "black box" model | SHAP explanations + feature importance reporting (Section 9) |

---

## 13. Deliverables Checklist

- [ ] EDA notebook with class imbalance findings
- [ ] Cleaned & feature-engineered dataset
- [ ] Imbalance-handling comparison (class_weight vs SMOTE)
- [ ] Logistic Regression baseline with full evaluation report
- [ ] Decision Tree / Random Forest / XGBoost comparison table
- [ ] Final selected model with justification (not based on accuracy alone)
- [ ] SHAP-based interpretability report
- [ ] Business cost-benefit summary
- [ ] FastAPI inference endpoint
- [ ] Streamlit dashboard for business users
- [ ] README + reproducibility instructions

---

## 14. Requirements File (Draft)

```
pandas
numpy
scikit-learn
imbalanced-learn
xgboost
lightgbm
shap
matplotlib
seaborn
fastapi
uvicorn
streamlit
joblib
```

---

*This plan should be treated as a living document — update the model comparison table, metrics, and roadmap as experimentation progresses.*