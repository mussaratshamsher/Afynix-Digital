import streamlit as st
import pandas as pd
import joblib
import os
import sys
from src.preprocessing import clean_data
from src.feature_engineering import engineer_features
from src.llm_insights import generate_insights

# Get directory containing app.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Add src to path
sys.path.append(os.path.join(BASE_DIR, 'src'))

st.set_page_config(page_title="Karachi Real Estate AI", layout="wide")

st.title("🏙️ Karachi Real Estate Pricing Prediction & AI Insights")
st.markdown("---")

# Load Data
@st.cache_data
def load_data():
    df = pd.read_csv(os.path.join(BASE_DIR, 'data', 'housing.csv'))
    return df

df_raw = load_data()
df_clean = clean_data(df_raw.copy())
df_engineered = engineer_features(df_clean.copy())

# Sidebar Navigation
page = st.sidebar.selectbox("Go to", ["Overview & EDA", "Model Performance", "Price Prediction", "AI Business Insights"])

if page == "Overview & EDA":
    st.header("📊 Dataset Overview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Records", len(df_raw))
    col2.metric("Locations", df_raw['location'].nunique())
    col3.metric("Avg Price (PKR)", f"{df_raw['price_pkr'].mean():,.0f}")
    
    st.subheader("Raw Data Sample")
    st.dataframe(df_raw.head())
    
    st.subheader("Price Distribution by Location")
    import plotly.express as px
    fig = px.box(df_raw, x="location", y="price_pkr", color="property_type", title="Property Prices across Karachi")
    st.plotly_chart(fig, use_container_width=True)

elif page == "Model Performance":
    st.header("📈 Model Evaluation")
    
    rf_model_path = os.path.join(BASE_DIR, 'models', 'random_forest.pkl')
    if os.path.exists(rf_model_path):
        # Here we would normally run src/evaluate.py and show results
        # For simplicity, we can show the saved images
        st.subheader("Actual vs Predicted (Random Forest)")
        rf_pred_img = os.path.join(BASE_DIR, 'notebooks', 'random_forest_actual_vs_pred.png')
        if os.path.exists(rf_pred_img):
            st.image(rf_pred_img)
        
        st.subheader("Feature Importance")
        feat_imp_img = os.path.join(BASE_DIR, 'notebooks', 'feature_importance.png')
        if os.path.exists(feat_imp_img):
            st.image(feat_imp_img)
    else:
        st.warning("Please run the training script first to see performance.")

elif page == "Price Prediction":
    st.header("🏠 Predict Property Price")
    
    col1, col2 = st.columns(2)
    with col1:
        location = st.selectbox("Select Location", df_raw['location'].unique())
        prop_type = st.selectbox("Property Type", df_raw['property_type'].unique())
        area = st.number_input("Area (Square Yards)", min_value=80, max_value=2000, value=120)
    
    with col2:
        bedrooms = st.slider("Bedrooms", 1, 10, 3)
        bathrooms = st.slider("Bathrooms", 1, 10, 2)
    
    if st.button("Predict Price"):
        rf_model_path = os.path.join(BASE_DIR, 'models', 'random_forest.pkl')
        model = joblib.load(rf_model_path)
        
        input_df = pd.DataFrame({
            'location': [location],
            'area_sq_yards': [area],
            'bedrooms': [bedrooms],
            'bathrooms': [bathrooms],
            'property_type': [prop_type],
            'room_density': [(bedrooms + bathrooms) / area]
        })
        
        prediction = model.predict(input_df)[0]
        st.success(f"Estimated Price: **PKR {prediction:,.0f}**")
        
        # AI Advisor
        st.markdown("---")
        st.subheader("🤖 AI Property Advisor")
        with st.spinner("Generating AI advice..."):
            advice_prompt = f"A {prop_type} in {location} with {area} sq yards, {bedrooms} beds, and {bathrooms} baths."
            # We can use a simplified call here or integrate with llm_insights
            advice = generate_insights(f"Property: {advice_prompt}", f"Predicted Price: {prediction}", "Location, Area")
            st.write(advice)

elif page == "AI Business Insights":
    st.header("💡 AI-Generated Business Insights")
    if st.button("Generate Strategy Report"):
        with st.spinner("Analyzing market data with Llama-3.3..."):
            summary = df_raw.describe().to_string()
            # Placeholder for results
            results = "R2: 0.88, MAE: 5M PKR"
            top_features = "Area, DHA Location, Property Type"
            insights = generate_insights(summary, results, top_features)
            st.markdown(insights)
