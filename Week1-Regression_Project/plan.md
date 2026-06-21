You are a Senior Machine Learning Engineer, Data Scientist, and MLOps Expert.

Your task is to build a complete end-to-end Machine Learning Regression project for Week 1 Assignment.

==================================================
PROJECT TITLE
==================================================

Real Estate Pricing Prediction for a Startup

Goal:
Predict house/property prices using regression models and generate business insights for a startup.

==================================================
OBJECTIVES
==================================================

1. Perform professional Exploratory Data Analysis (EDA)
2. Clean and preprocess data
3. Engineer useful features
4. Train multiple regression models
5. Compare model performance
6. Detect overfitting
7. Explain feature importance
8. Generate business recommendations
9. Build a Streamlit dashboard
10. Deploy-ready project structure

==================================================
TECH STACK
==================================================

Language:
- Python

Libraries:
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- plotly
- streamlit
- joblib

LLM Integration:
- Groq Free Tier
- Model:
    llama-3.3-70b-versatile

Alternative:
    gemma2-9b-it

Use LLM only for:
- Automated insight generation
- Dataset summary
- Business recommendations

Do NOT use paid APIs.

==================================================
PROJECT STRUCTURE
==================================================

real-estate-regression/
│
├── data/
│   └── housing.csv
│
├── notebooks/
│   └── eda.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── train.py
│   ├── evaluate.py
│   ├── explainability.py
│   └── llm_insights.py
│
├── models/
│   ├── linear_regression.pkl
│   └── random_forest.pkl
│
├── app.py
│
├── requirements.txt
│
└── README.md

==================================================
STEP 1 — DATA INSPECTION
==================================================

Load dataset and display:

- shape
- columns
- head()
- info()
- describe()

Check:

- missing values
- duplicates
- categorical columns
- numerical columns

Generate summary table.

==================================================
STEP 2 — EDA
==================================================

Create professional visualizations.

1. Correlation Heatmap

2. Price Distribution

3. Area vs Price Scatter Plot

4. Bedrooms vs Price

5. Bathrooms vs Price

6. Boxplots for Outlier Detection

7. Pairplots for Key Features

8. Feature Correlation Ranking

Generate insights below every chart.

==================================================
STEP 3 — DATA CLEANING
==================================================

Handle missing values:

Numerical:
- Median imputation

Categorical:
- Mode imputation

Remove duplicates.

Handle outliers using:

- IQR Method

Create reusable preprocessing pipeline.

==================================================
STEP 4 — FEATURE ENGINEERING
==================================================

Perform:

1. Label Encoding
2. One Hot Encoding

Create new features if applicable:

Examples:

- price_per_sqft
- room_density
- bathrooms_per_room

Scale features using:

StandardScaler

Create sklearn Pipeline.

==================================================
STEP 5 — TRAIN TEST SPLIT
==================================================

Use:

test_size=0.2

random_state=42

==================================================
STEP 6 — MODEL TRAINING
==================================================

Train Model 1:

Linear Regression

Purpose:
Baseline model

Train Model 2:

Random Forest Regressor

Use GridSearchCV for tuning.

Tune:

- n_estimators
- max_depth
- min_samples_split

==================================================
STEP 7 — EVALUATION
==================================================

Calculate:

- R² Score
- MAE
- MSE
- RMSE

Create comparison table.

Visualizations:

1. Actual vs Predicted
2. Residual Plot
3. Error Distribution

Determine:

- Best Model
- Worst Model

==================================================
STEP 8 — OVERFITTING ANALYSIS
==================================================

Compare:

Train Metrics
vs
Test Metrics

Generate report:

IF train score >> test score

Explain:
- Why overfitting occurred
- Recommended fixes

==================================================
STEP 9 — FEATURE IMPORTANCE
==================================================

For Random Forest:

Display:

- Top 10 Important Features

Visualize using bar chart.

Generate interpretation:

Which factors influence price most?

Explain in simple business language.

==================================================
STEP 10 — AI INSIGHT GENERATION
==================================================

Using Groq Free Tier:

Generate:

1. Dataset Summary
2. Top Pricing Drivers
3. Risks in Dataset
4. Startup Recommendations
5. Business Decisions Based on Results

Output should be concise and professional.

==================================================
STEP 11 — STREAMLIT DASHBOARD
==================================================

Create dashboard with:

Page 1:
Dataset Overview

Page 2:
EDA

Page 3:
Model Performance

Page 4:
Feature Importance

Page 5:
Price Prediction

User inputs:

- area
- bedrooms
- bathrooms
- location
- etc

Predict property price.

==================================================
STEP 12 — README
==================================================

Generate professional README containing:

- Problem Statement
- Dataset
- Workflow
- Models Used
- Results
- Future Improvements

==================================================
BONUS (HIGHLY RECOMMENDED)
==================================================

Create:

AI Property Advisor

User enters property details.

System predicts price.

Groq LLM generates:

- Property valuation explanation
- Market insight
- Investment recommendation

==================================================
DELIVERABLES
==================================================

Provide:

1. Complete source code
2. Modular architecture
3. Streamlit application
4. Trained model files
5. requirements.txt
6. README.md
7. Deployment instructions

Write production-quality code with comments, error handling, reusable functions, and clean architecture.