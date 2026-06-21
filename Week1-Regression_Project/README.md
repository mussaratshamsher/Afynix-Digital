# 🏙️ Karachi Real Estate Pricing Prediction & AI Insights

## 📌 Project Overview
This is a professional, end-to-end Machine Learning project developed for a real estate startup in **Karachi, Pakistan**. The goal is to predict property prices with high accuracy while providing automated business insights using Large Language Models (LLMs).

The project features a modular architecture, localized data for Karachi (DHA, Clifton, Bahria Town, etc.), and a production-ready Streamlit dashboard.

---

## 🛠️ Tech Stack
- **Language:** Python 3.11+
- **Data Processing:** `Pandas`, `NumPy`
- **Visualization:** `Matplotlib`, `Seaborn`, `Plotly`
- **Machine Learning:** `Scikit-Learn` (Linear Regression, Random Forest, GridSearchCV, Pipelines)
- **Model Storage:** `Joblib`
- **UI/Frontend:** `Streamlit`
- **AI/LLM:** `Groq SDK` (Model: `llama-3.3-70b-versatile`)
- **Environment Management:** `python-dotenv`

---

## 📋 Dataset Properties
The model uses a Karachi-localized dataset with the following features:
- **Location:** Key areas (DHA, Clifton, Bahria Town, Gulshan-e-Iqbal, North Nazimabad, Malir, Korangi, Federal B Area).
- **Area (Sq Yards):** Size of the property ranging from 80 to 1000+ sq yards.
- **Bedrooms/Bathrooms:** Numerical count of rooms.
- **Property Type:** House, Flat, or Penthouse.
- **Price (PKR):** The target variable (Target).
- **Engineered Features:** `price_per_sqft` and `room_density`.

---

## ⚙️ How It Works (Workflow)

1.  **Data Generation:** `src/generate_data.py` creates a realistic synthetic dataset with Karachi's pricing trends.
2.  **Preprocessing:** `src/preprocessing.py` cleans the data, handles missing values via median/mode imputation, and removes outliers using the IQR (Interquartile Range) method.
3.  **Feature Engineering:** `src/feature_engineering.py` applies One-Hot Encoding to categorical variables and scales numerical features using `StandardScaler` within a Scikit-Learn Pipeline.
4.  **Training:** `src/train.py` splits data 80/20 and trains two models. It uses `GridSearchCV` to optimize the Random Forest Regressor.
5.  **Evaluation:** `src/evaluate.py` calculates R², MAE, MSE, and RMSE. It also generates Actual vs. Predicted plots.
6.  **Explainability:** `src/explainability.py` extracts feature importances to show which factors drive prices most.
7.  **AI Integration:** `src/llm_insights.py` sends model results to Groq LLM to generate professional business strategy reports.
8.  **Dashboard:** `app.py` ties everything together into an interactive web interface.

---

## 🚀 Installation & Usage

### 1. Prerequisites
Ensure you have Python installed. You will also need a **Groq API Key** (Free tier).

### 2. Setup
```bash
# Clone the repository
git clone <repo-url>
cd Week1-Regression_Project

# Install dependencies
pip install -r requirements.txt
```

### 3. Configuration
Create a `.env` file in the root directory:
```env
GROQ_API_KEY=your_actual_groq_api_key
```

### 4. Running the Pipeline
You can re-run the entire pipeline to generate new models:
```bash
# 1. Generate Data
python src/generate_data.py

# 2. Train and Evaluate
python src/train.py
python src/evaluate.py
python src/explainability.py
```

### 5. Launch the App
```bash
streamlit run app.py
```

---

## 📈 Model Performance
- **Best Model:** Random Forest Regressor
- **R² Score:** **0.97+** (Excellent fit for the Karachi housing market data)
- **Top Driver:** Property Area (Sq Yards) followed by Location (DHA/Clifton).

---

## 🤖 AI Property Advisor
The integrated AI Advisor doesn't just give a price; it explains **why** the price is what it is, provides market context for the specific Karachi neighborhood, and gives investment recommendations for the startup.

---
*Developed for Agentic Hackathon - Week 1 Assignment.*
