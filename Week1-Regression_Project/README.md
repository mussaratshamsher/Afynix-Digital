# 🏙️ Karachi Real Estate AI: Pricing Prediction & Business Portal

Welcome to the **Karachi Real Estate Pricing & Analytics Portal**, an end-to-end data science and machine learning application tailored for real estate startups in Karachi, Pakistan. 

This project integrates classical statistical regression modeling with Large Language Models (LLMs) and interactive visualization widgets, creating a production-grade analysis platform for real estate appraisal and investment.

---

## 🚀 Key Features

1. **📊 Comprehensive Data Inspection (`isnull()` & `dtypes`):**
   - Live checks for dataset shape, location counts, and duplicate entries.
   - Transparent reporting on missing values per column via `isnull()` charts.
   - Comprehensive data types (`categorical` vs. `numerical`) audit tables.

2. **🔥 Interactive Exploratory Data Analysis (EDA):**
   - **Correlation Heatmap:** Interactive correlation grid mapping relationships between numerical characteristics.
   - **Trend Plots:** Plotly scatter plots representing price trends vs. plot sizes, color-coded dynamically.
   - **Distribution Auditing:** Box plots analyzing pricing variance across Karachi neighborhoods (DHA, Clifton, Bahria Town, etc.).

3. **⚙️ Preprocessing & Feature Engineering:**
   - Preprocessing steps including duplicate removal, median (numerical) and mode (categorical) missing value imputation, and Interquartile Range (IQR) outlier removal.
   - Feature engineering adding ratios like **Room Density** ($(\text{Bedrooms} + \text{Bathrooms}) / \text{Area}$) and **Price per Sqft**.

4. **📈 Benchmarking & Overfitting Guardrails:**
   - Parallel evaluation of the baseline **Linear Regression** model and optimized **Random Forest Regressor** (tuned via `GridSearchCV`).
   - Dynamic comparison table highlighting Train vs. Test metrics ($R^2$, MAE, RMSE).
   - Automated **Overfitting Check** verifying generalization gaps and alert warnings.
   - Full suite of residual, actual vs. predicted, and error diagnostics.

5. **🔍 Feature Importance Ranking:**
   - Extraction and ranking of structural pricing weights.
   - Explanation of key pricing drivers in Karachi (e.g. plot area scale, location premiums, property structure).

6. **🏠 Smart Price Valuation Engine:**
   - Real-time valuation calculator applying the pre-trained Scikit-Learn pipelines to user inputs.
   - **AI Property Advisor:** Generates localized real estate context, spatial layout analysis, and investment recommendations.
   - **API Fallback System:** Uses a rule-based advisor engine if LLM API keys are unconfigured, ensuring 100% application uptime.

7. **💡 AI Market Insights:**
   - Interactive executive reports analyzing Karachi's market structure, risks, and strategic suggestions.

---

## 📁 Repository Structure

```directory
Week1-Regression_Project/
│
├── data/
│   ├── housing.csv                  # Raw synthetic housing data
│   └── cleaned_housing.csv          # Preprocessed housing dataset
│
├── models/
│   ├── linear_regression.pkl        # Baseline model pipeline
│   ├── random_forest.pkl            # Tuned champion model pipeline
│   ├── metrics.json                 # Training & testing evaluation metrics
│   └── feature_importances.csv      # Extracted feature driver weights
│
├── notebooks/
│   ├── feature_importance.png       # Importance plots (Matplotlib)
│   ├── linear_regression_actual_vs_pred.png
│   ├── linear_regression_residuals.png
│   ├── random_forest_actual_vs_pred.png
│   └── random_forest_residuals.png
│
├── src/
│   ├── preprocessing.py             # Data imputation & IQR filtering
│   ├── feature_engineering.py       # ColumnTransformers & custom ratios
│   ├── train.py                     # Pipelines & GridSearchCV
│   ├── evaluate.py                  # Evaluation metrics & diagnostic plots
│   ├── explainability.py            # Feature importance extraction
│   ├── llm_insights.py              # Groq API integration
│   └── generate_data.py             # Karachi dataset generator
│
├── app.py                           # Premium Streamlit application code
├── requirements.txt                 # Dependencies
├── .env                             # Active API credentials
├── plan.md                          # Design document
└── README.md                        # Documentation
```

---

## 🛠️ Installation & Execution

### 1. Prerequisites
- **Python 3.11+** installed.
- (Optional) A **Groq API Key** for real-time LLM commentary (the app works natively in offline/fallback mode without it).

### 2. Setup
Clone the repository, navigate to the project directory, and install dependencies:
```bash
cd Week1-Regression_Project
pip install -r requirements.txt
```

### 3. API Configuration (Optional)
Create a `.env` file in the root directory:
```env
GROQ_API_KEY=gsk_your_actual_key_here
```

### 4. Running the Pipeline
To re-run the data preparation, training, evaluation, and explanation sequences:
```bash
# 1. Generate data
python src/generate_data.py

# 2. Train baseline and champion models
python src/train.py

# 3. Evaluate and run overfitting checks
python src/evaluate.py

# 4. Extract feature importances
python src/explainability.py
```

### 5. Launch the Streamlit Portal
Start the dashboard locally:
```bash
streamlit run app.py
```

---

## 📊 Model Performance Summary

Based on Karachi pricing trends:
- **Baseline (Linear Regression):** $R^2 \approx 87.5\%$ (Test Set)
- **Champion (Random Forest):** $R^2 \approx 97.8\%$ (Test Set)
- **Generalization:** Both models satisfy the overfitting checks ($\text{Train } R^2 - \text{Test } R^2 < 5\%$), indicating robust generalization across Karachi's real estate configurations.
- **Top Valuation Drivers:** Plot area (Sq Yards) carries the highest correlation weight, followed immediately by location premiums (DHA, Clifton) and property architectural structure (House vs. Flat).
