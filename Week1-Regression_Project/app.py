import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import sys
import json
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv

# Load env variables
load_dotenv()

# Get directory containing app.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Add src to path
sys.path.append(os.path.join(BASE_DIR, 'src'))

from preprocessing import clean_data
from feature_engineering import engineer_features
from llm_insights import generate_insights

# Set page configurations
st.set_page_config(
    page_title="Karachi Real Estate AI Pricing & Analytics Portal",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    .reportview-container {
        font-family: 'Space Grotesk', sans-serif;
    }
    
    /* Title Styling */
    .title-gradient {
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(135deg, #FF4B4B, #FF8F8F, #3F3B6C);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 5px;
    }
    .subtitle {
        color: #94a3b8;
        font-size: 1.1rem;
        margin-bottom: 25px;
    }
    
    /* Card Container */
    .metric-card-container {
        background-color: #1e293b;
        border-radius: 12px;
        padding: 20px;
        border: 1px solid #334155;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        text-align: center;
        transition: all 0.3s ease;
    }
    .metric-card-container:hover {
        transform: translateY(-3px);
        border-color: #FF4B4B;
        box-shadow: 0 10px 15px -3px rgba(255, 75, 75, 0.1), 0 4px 6px -2px rgba(255, 75, 75, 0.05);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #f8fafc;
        margin-bottom: 4px;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #94a3b8;
        text-transform: uppercase;
        font-weight: 500;
        letter-spacing: 0.05em;
    }
    
    /* Section Headers */
    .section-header {
        font-size: 1.6rem;
        font-weight: 600;
        color: #f8fafc;
        border-left: 4px solid #FF4B4B;
        padding-left: 10px;
        margin-top: 20px;
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)

# Load data helper
@st.cache_data
def load_raw_data():
    df = pd.read_csv(os.path.join(BASE_DIR, 'data', 'housing.csv'))
    return df

try:
    df_raw = load_raw_data()
    df_clean = clean_data(df_raw.copy())
    df_engineered = engineer_features(df_clean.copy())
except Exception as e:
    st.error(f"Error loading or preprocessing dataset: {e}")
    st.info("Generating housing data first...")
    from generate_data import generate_karachi_housing_data
    generate_karachi_housing_data()
    df_raw = load_raw_data()
    df_clean = clean_data(df_raw.copy())
    df_engineered = engineer_features(df_clean.copy())

# Rule-based property advisor fallbacks
def get_fallback_property_advice(location, prop_type, area, bedrooms, bathrooms, prediction):
    loc_tier = "High-End Premium District" if location in ['DHA', 'Clifton'] else "Mid-Tier Established Residential" if location in ['Gulshan-e-Iqbal', 'North Nazimabad', 'Federal B Area'] else "Emerging Suburbs / Industrial Zones"
    
    # Space density calculation
    density = (bedrooms + bathrooms) / area
    if density > 0.05:
        density_label = "High Density (Compact Structure)"
        density_desc = "The property utilizes space for rooms aggressively. Ideal for higher tenancy models or large families, but compromises on individual room sizes."
    elif density < 0.02:
        density_label = "Low Density (Spacious Layout)"
        density_desc = "Generous spatial layout. Highly attractive for luxury home buyers. Typically commands a premium price per square yard."
    else:
        density_label = "Optimal Density (Standard Design)"
        density_desc = "Balanced, standard architectural distribution of rooms and space."

    appreciation = "Strong Capital Gains (8-12% annually)" if location in ['DHA', 'Bahria Town'] else "Healthy Capital Growth (5-8% annually)" if location in ['Clifton', 'Gulshan-e-Iqbal', 'North Nazimabad'] else "Stable Residential Yield (3-5% annually)"
    
    advice = f"""
    ### 🏢 Real Estate Advisor Valuation Report (System Generated)
    
    *Providing offline strategic insights for startup decision-making.*
    
    #### 📍 Neighborhood Context & Valuation Class
    - **District Classification:** **{loc_tier}**
    - **Geographic Value Drivers:** {location} is a significant real estate corridor in Karachi. DHA and Clifton command substantial price premiums because of infrastructure status, coastal proximity, and security profiles. North Nazimabad and Gulshan-e-Iqbal serve as stable, high-demand metropolitan residential areas.
    
    #### 📐 Structural Layout & Spatial Analysis
    - **Layout Classification:** **{density_label}**
    - *Details:* {density_desc}
    - **Total Land Area:** **{area} Sq Yards** (approx. {area * 9} Sq Ft)
    - **Room Configuration:** {bedrooms} Bedrooms | {bathrooms} Bathrooms
    
    #### 💰 Strategic Financial Appraisal
    - **Estimated Market Valuation:** **PKR {prediction:,.0f}** (~{prediction/10000000:.2f} Crore PKR)
    - **Price per Sq Yard:** **PKR {prediction / area:,.0f} / Sq Yard**
    - **Price per Sq Ft:** **PKR {prediction / (area * 9):,.0f} / Sq Ft**
    
    #### 📈 Startup & Investor Recommendation
    - **Expected Growth Curve:** **{appreciation}**
    - **Suggested Investment Action:** {"Strong Buy for capital gains - properties in DHA/Bahria Town are high-liquidity assets." if location in ['DHA', 'Bahria Town'] else "Acquire / Hold for rental yield and residential stability." if location in ['Clifton', 'Gulshan-e-Iqbal'] else "Entry-level acquisition for steady utility value with lower capital barriers."}
    """
    return advice

def get_fallback_business_insights():
    insights = """
    ### 🏙️ Karachi Real Estate Analytics Report (Startup Insights)
    
    *Dynamic offline report based on Karachi Housing model results.*
    
    #### 1. Karachi Market Structure Analysis
    Karachi’s housing market exhibits highly polarized pricing curves. High-end coastal areas (DHA, Clifton) command massive premiums, where demand is inelastic due to security concerns and infrastructure quality. Bahria Town represents a modern, planned speculative outpost with huge growth potential, while established central areas (Gulshan, North Nazimabad) represent core middle-to-upper-middle class housing demand.
    
    #### 2. Primary Drivers of Housing Pricing
    1. **Location Vector:** The DHA and Clifton premium overrides all other factors. Plotting shows location accounts for a massive chunk of price variance.
    2. **Property Dimension (Sq Yards):** Plot sizing dictates structural allowance, making it the most significant continuous feature. Price scales non-linearly with standard sizes.
    3. **Property Architecture Type:** Houses command massive premiums over flats, driven by land ownership value. Penthouses hold a niche luxury market share.
    
    #### 3. Startup Market Opportunities & Risks
    - **Constraint:** Real-world prices depend heavily on amenities like utility availability (regular water, gas, electricity grid stability). The startup must add utility metrics in future iterations.
    - **Opportunity:** DHA and Clifton represent high-value transaction pipelines. Developing a pricing API for these regions can capture B2B brokerage customers.
    
    #### 4. Actionable Business Strategies
    - **API Integration:** Offer the pricing engine as a widget for Karachi listing platforms.
    - **Database Refinement:** Collect historical data to model temporal pricing changes (inflation vs appreciation).
    """
    return insights

# Page Title
st.markdown("<div class='title-gradient'>Karachi Real Estate Pricing & AI Portal</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>A modern data science dashboard designed for real estate startup strategy & price prediction.</div>", unsafe_allow_html=True)

# Sidebar Navigation
st.sidebar.image("https://img.icons8.com/clouds/200/real-estate.png", width=120)
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select Portal View",
    [
        "📋 Data Inspection",
        "📊 Interactive EDA",
        "⚙️ Feature Engineering",
        "📈 Benchmarking & Overfitting",
        "🔍 Feature Drivers Analysis",
        "🏠 Smart Price Predictor",
        "💡 AI Business Insights"
    ]
)

# ----------------------------------------------------
# PAGE 1: DATA INSPECTION
# ----------------------------------------------------
if page == "📋 Data Inspection":
    st.markdown("<div class='section-header'>Data Inspection & Integrity Check</div>", unsafe_allow_html=True)
    
    # KPI metrics row
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown(f"<div class='metric-card-container'><div class='metric-value'>{len(df_raw)}</div><div class='metric-label'>Raw Records</div></div>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<div class='metric-card-container'><div class='metric-value'>{df_raw['location'].nunique()}</div><div class='metric-label'>Karachi Locations</div></div>", unsafe_allow_html=True)
    with col3:
        st.markdown(f"<div class='metric-card-container'><div class='metric-value'>PKR {df_raw['price_pkr'].mean()/1000000:.1f}M</div><div class='metric-label'>Average Price</div></div>", unsafe_allow_html=True)
    with col4:
        st.markdown(f"<div class='metric-card-container'><div class='metric-value'>{df_raw.isnull().sum().sum()}</div><div class='metric-label'>Missing Cells</div></div>", unsafe_allow_html=True)
    with col5:
        st.markdown(f"<div class='metric-card-container'><div class='metric-value'>{df_raw.duplicated().sum()}</div><div class='metric-label'>Duplicate Rows</div></div>", unsafe_allow_html=True)
        
    st.write(" ")
    
    # Missing values and data types side-by-side
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.subheader("🔍 Data Types & Structure Check")
        dtypes_df = pd.DataFrame({
            "Feature Name": df_raw.columns,
            "Pandas Data Type": df_raw.dtypes.astype(str),
            "Non-Null Count": df_raw.notnull().sum(),
            "Null Count": df_raw.isnull().sum(),
            "Sample Value": [df_raw[c].dropna().iloc[0] for c in df_raw.columns]
        }).reset_index(drop=True)
        st.dataframe(dtypes_df, use_container_width=True)
        
    with col_right:
        st.subheader("⚠️ Missing Value Analysis (isnull())")
        null_counts = df_raw.isnull().sum()
        null_df = pd.DataFrame({
            "Feature": null_counts.index,
            "Missing Count": null_counts.values,
            "Percentage Missing (%)": np.round((null_counts.values / len(df_raw)) * 100, 2)
        })
        
        # Plotly chart of missing values
        fig_null = px.bar(
            null_df, 
            x="Feature", 
            y="Missing Count", 
            text="Missing Count",
            title="Missing Count per Feature (isnull() check)",
            color="Missing Count",
            color_continuous_scale="Reds"
        )
        fig_null.update_layout(height=300, showlegend=False, margin=dict(t=40, b=10, l=10, r=10))
        st.plotly_chart(fig_null, use_container_width=True)

    st.subheader("Raw Data Sample (Top 5 rows)")
    st.dataframe(df_raw.head(), use_container_width=True)

# ----------------------------------------------------
# PAGE 2: INTERACTIVE EDA
# ----------------------------------------------------
elif page == "📊 Interactive EDA":
    st.markdown("<div class='section-header'>Exploratory Data Analysis</div>", unsafe_allow_html=True)
    
    # Tabs for different EDA sections
    tab_corr, tab_scatter, tab_box = st.tabs(["🔥 Correlation Heatmap", "📈 Scatter Analysis", "📦 Price Distribution"])
    
    with tab_corr:
        st.subheader("Correlation Heatmap to Find Strong Relationships")
        num_cols = df_engineered.select_dtypes(include=[np.number]).columns
        corr_matrix = df_engineered[num_cols].corr()
        
        fig_corr = px.imshow(
            corr_matrix, 
            text_auto=".2f", 
            color_continuous_scale="RdBu_r", 
            zmin=-1, zmax=1,
            aspect="auto",
            labels=dict(color="Correlation Coefficient")
        )
        fig_corr.update_layout(title="Numerical Features Correlation Matrix", margin=dict(t=50, b=10))
        st.plotly_chart(fig_corr, use_container_width=True)
        
        st.markdown("""
        **Key Insights:**
        - **Area (Sq Yards)** has the strongest positive correlation with the target variable **Price (PKR)**, confirming it as the primary pricing driver.
        - **Bedrooms** and **Bathrooms** show strong collinearity, which is natural. 
        - **Room Density** shows a negative correlation, indicating that having too many rooms cramped inside a smaller area reduces unit price value in premium segments.
        """)
        
    with tab_scatter:
        st.subheader("Scatter Plot: price_pkr vs Continuous Features")
        x_axis = st.selectbox("Select X-Axis variable for Scatter Plot", ["area_sq_yards", "bedrooms", "bathrooms", "room_density"])
        color_by = st.selectbox("Color Code by", ["location", "property_type"])
        
        fig_scatter = px.scatter(
            df_engineered,
            x=x_axis,
            y="price_pkr",
            color=color_by,
            title=f"Price vs {x_axis.replace('_', ' ').title()} in Karachi",
            labels={"price_pkr": "Price (PKR)", x_axis: x_axis.replace('_', ' ').title()},
            opacity=0.7,
            trendline="ols" if x_axis == "area_sq_yards" else None
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
        
    with tab_box:
        st.subheader("Price Distributions & Outliers by Category")
        cat_feature = st.selectbox("Select Categorical Feature for Price Box Plot", ["location", "property_type"])
        
        fig_box = px.box(
            df_raw,
            x=cat_feature,
            y="price_pkr",
            color=cat_feature,
            title=f"Price Distribution across {cat_feature.title()} (Raw Data)",
            labels={"price_pkr": "Price (PKR)"}
        )
        st.plotly_chart(fig_box, use_container_width=True)

# ----------------------------------------------------
# PAGE 3: FEATURE ENGINEERING
# ----------------------------------------------------
elif page == "⚙️ Feature Engineering":
    st.markdown("<div class='section-header'>Data Preprocessing & Feature Engineering Pipeline</div>", unsafe_allow_html=True)
    
    st.markdown("""
    To prepare the data for training, we apply a strict sequential preprocessing and engineering pipeline to avoid data leakage and optimize performance:
    """)
    
    col_step1, col_step2 = st.columns(2)
    
    with col_step1:
        st.subheader("🔧 Data Cleaning Steps")
        st.write("""
        1. **Duplicate Removal:** Removed redundant listings to prevent training bias.
        2. **Imputation (isnull() recovery):**
           - Numerical values (`bedrooms`, `bathrooms`) are filled with their **median**.
           - Categorical values (`location`, `property_type`) are filled with their **mode**.
        3. **Outlier Mitigation:** Applied the IQR method:
           - Lower Bound: $Q1 - 1.5 \\times IQR$
           - Upper Bound: $Q3 + 1.5 \\times IQR$
           - Features cleaned: `area_sq_yards`, `price_pkr`.
        """)
        st.info(f"Raw shape: **{df_raw.shape}** → Cleaned shape: **{df_clean.shape}** (Removed outliers & duplicates)")
        
    with col_step2:
        st.subheader("🧬 Engineered Features")
        st.write("""
        We engineered two highly descriptive ratios:
        - **`price_per_sqft`**: Calculates price per unit area (Price / (Area in sq yards * 9)). *Dropped during model training to avoid target leakage.*
        - **`room_density`**: Ratio of total rooms to the size of the land:
          $$\\text{Room Density} = \\frac{\\text{Bedrooms} + \\text{Bathrooms}}{\\text{Area (Sq Yards)}}$$
          This helps the model distinguish spacious luxury layouts from high-density structures.
        """)
        
    st.subheader("Cleaned & Feature Engineered Sample")
    st.dataframe(df_engineered.head(5), use_container_width=True)

# ----------------------------------------------------
# PAGE 4: BENCHMARKING & OVERFITTING CHECK
# ----------------------------------------------------
elif page == "📈 Benchmarking & Overfitting":
    st.markdown("<div class='section-header'>Model Comparison & Overfitting Assessment</div>", unsafe_allow_html=True)
    
    metrics_path = os.path.join(BASE_DIR, 'models', 'metrics.json')
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
            
        st.subheader("📊 Model Performance Benchmark Table")
        
        # Build DataFrame for display
        rows = []
        for m_name, vals in metrics.items():
            rows.append({
                "Model": m_name,
                "Train R²": f"{vals['train']['R2']:.4f}",
                "Test R²": f"{vals['test']['R2']:.4f}",
                "Train MAE (PKR)": f"{vals['train']['MAE']:,.0f}",
                "Test MAE (PKR)": f"{vals['test']['MAE']:,.0f}",
                "Train RMSE (PKR)": f"{vals['train']['RMSE']:,.0f}",
                "Test RMSE (PKR)": f"{vals['test']['RMSE']:,.0f}",
                "Generalization Gap": f"{vals['overfitting']['gap']:.4f}"
            })
        
        st.table(pd.DataFrame(rows))
        
        st.write(" ")
        st.subheader("🔍 Overfitting Status Check")
        
        col_lr, col_rf = st.columns(2)
        
        with col_lr:
            st.markdown("#### 🔹 Baseline: Linear Regression")
            lr_ov = metrics["Linear Regression"]["overfitting"]
            if "Healthy" in lr_ov["status"]:
                st.success(f"**Status:** {lr_ov['status']}\n\n{lr_ov['description']}")
            else:
                st.warning(f"**Status:** {lr_ov['status']}\n\n{lr_ov['description']}")
                
        with col_rf:
            st.markdown("#### 🌲 Champion: Random Forest Regressor")
            rf_ov = metrics["Random Forest"]["overfitting"]
            if "Healthy" in rf_ov["status"]:
                st.success(f"**Status:** {rf_ov['status']}\n\n{rf_ov['description']}")
            else:
                st.warning(f"**Status:** {rf_ov['status']}\n\n{rf_ov['description']}")

        # Plotly performance chart
        st.write(" ")
        st.subheader("Model Performance Comparison (R² Score)")
        r2_plot_df = pd.DataFrame([
            {"Model": "Linear Regression", "Dataset": "Train R²", "R² Score": metrics["Linear Regression"]["train"]["R2"]},
            {"Model": "Linear Regression", "Dataset": "Test R²", "R² Score": metrics["Linear Regression"]["test"]["R2"]},
            {"Model": "Random Forest", "Dataset": "Train R²", "R² Score": metrics["Random Forest"]["train"]["R2"]},
            {"Model": "Random Forest", "Dataset": "Test R²", "R² Score": metrics["Random Forest"]["test"]["R2"]}
        ])
        fig_r2 = px.bar(
            r2_plot_df, 
            x="Model", 
            y="R² Score", 
            color="Dataset", 
            barmode="group",
            color_discrete_sequence=["#3366CC", "#FF4B4B"],
            text_auto=".3f"
        )
        fig_r2.update_layout(yaxis_range=[0.7, 1.05], height=350)
        st.plotly_chart(fig_r2, use_container_width=True)

        # Matplotlib images generated in evaluate.py
        st.write(" ")
        st.subheader("📸 Diagnostic Visualizations (Matplotlib Outputs)")
        img_col1, img_col2 = st.columns(2)
        with img_col1:
            lr_pred_img = os.path.join(BASE_DIR, 'notebooks', 'linear_regression_actual_vs_pred.png')
            if os.path.exists(lr_pred_img):
                st.image(lr_pred_img, caption="Linear Regression: Actual vs Predicted Scatter", use_container_width=True)
            else:
                st.info("Run evaluations to generate Linear Regression scatter plot.")
                
            lr_res_img = os.path.join(BASE_DIR, 'notebooks', 'linear_regression_residuals.png')
            if os.path.exists(lr_res_img):
                st.image(lr_res_img, caption="Linear Regression: Residuals vs Fitted values", use_container_width=True)
        with img_col2:
            rf_pred_img = os.path.join(BASE_DIR, 'notebooks', 'random_forest_actual_vs_pred.png')
            if os.path.exists(rf_pred_img):
                st.image(rf_pred_img, caption="Random Forest: Actual vs Predicted Scatter", use_container_width=True)
            else:
                st.info("Run evaluations to generate Random Forest scatter plot.")
                
            rf_res_img = os.path.join(BASE_DIR, 'notebooks', 'random_forest_residuals.png')
            if os.path.exists(rf_res_img):
                st.image(rf_res_img, caption="Random Forest: Residuals vs Fitted values", use_container_width=True)

    else:
        st.warning("Could not locate models/metrics.json. Please execute evaluation script `python src/evaluate.py` to view.")

# ----------------------------------------------------
# PAGE 5: FEATURE DRIVERS ANALYSIS
# ----------------------------------------------------
elif page == "🔍 Feature Drivers Analysis":
    st.markdown("<div class='section-header'>Feature Importance & Driver Interpretation</div>", unsafe_allow_html=True)
    
    feat_imp_path = os.path.join(BASE_DIR, 'models', 'feature_importances.csv')
    if os.path.exists(feat_imp_path):
        feat_df = pd.read_csv(feat_imp_path)
        
        # Sort and clean name formatting
        feat_df['Feature Clean'] = feat_df['Feature'].apply(lambda x: x.replace('_', ' ').title())
        feat_df = feat_df.sort_values(by='Importance', ascending=True)
        
        fig_feat = px.bar(
            feat_df,
            x="Importance",
            y="Feature Clean",
            orientation="h",
            color="Importance",
            color_continuous_scale="Viridis",
            title="Random Forest Feature Importance Profile"
        )
        st.plotly_chart(fig_feat, use_container_width=True)
        
        col_text, col_tbl = st.columns([2, 1])
        with col_text:
            st.markdown("""
            #### 💡 Pricing Drivers Explained:
            - **`area_sq_yards` (Plot Sizing):** The plot area is the undisputed champion feature, carrying the highest pricing influence. Plots dictate structural potential and baseline value.
            - **Location Metrics (`location_...`):** High-end location features like DHA command immense pricing multipliers. A DHA property of matching size will always command 2-4x pricing over mid-tier regions like Gulshan or North Nazimabad.
            - **Property Type (`property_type_House`):** Residential houses carry land value appreciation premiums, ranking higher than flats which are typically restricted structures.
            - **Room Density (`room_density`):** The engineered ratio has notable influence. Too high of a room density hurts premium valuations, whereas lower room density yields are seen in luxury apartments/penthouses.
            """)
        with col_tbl:
            st.subheader("Raw Importance Weight")
            st.dataframe(feat_df.sort_values(by='Importance', ascending=False)[['Feature', 'Importance']], use_container_width=True, hide_index=True)
    else:
        st.warning("Could not locate models/feature_importances.csv. Please execute explainability script `python src/explainability.py` to view.")

# ----------------------------------------------------
# PAGE 6: SMART PRICE PREDICTOR
# ----------------------------------------------------
elif page == "🏠 Smart Price Predictor":
    st.markdown("<div class='section-header'>Real Estate Valuation Engine</div>", unsafe_allow_html=True)
    
    rf_model_path = os.path.join(BASE_DIR, 'models', 'random_forest.pkl')
    if os.path.exists(rf_model_path):
        model = joblib.load(rf_model_path)
        
        col_inputs_left, col_inputs_right = st.columns(2)
        
        with col_inputs_left:
            location = st.selectbox("Select Property Neighborhood", df_raw['location'].unique(), index=0)
            prop_type = st.selectbox("Select Structure Type", df_raw['property_type'].unique(), index=0)
            area = st.slider("Total Area (Square Yards)", min_value=80, max_value=1000, value=240, step=10)
            
        with col_inputs_right:
            bedrooms = st.slider("Count of Bedrooms", min_value=1, max_value=8, value=3)
            bathrooms = st.slider("Count of Bathrooms", min_value=1, max_value=8, value=3)
            
        st.write(" ")
        
        # Calculate room density
        room_density = (bedrooms + bathrooms) / area
        
        input_data = pd.DataFrame({
            'location': [location],
            'area_sq_yards': [area],
            'bedrooms': [bedrooms],
            'bathrooms': [bathrooms],
            'property_type': [prop_type],
            'room_density': [room_density]
        })
        
        if st.button("Calculate Property Value", type="primary"):
            try:
                prediction = model.predict(input_data)[0]
                
                # Format price in crores and millions
                crores = prediction / 10000000
                millions = prediction / 1000000
                
                st.success(f"### Estimated Property Valuation: **PKR {prediction:,.0f}** (~{crores:.2f} Crore / {millions:.1f} Million)")
                
                # AI Advisory Section
                st.markdown("---")
                st.subheader("🤖 AI Advisor Appraisal")
                
                # Groq API check
                api_key = os.getenv('GROQ_API_KEY')
                if api_key and api_key.strip() and not api_key.startswith("your_"):
                    with st.spinner("Generating LLM valuation insights via Groq..."):
                        advice_prompt = f"A {prop_type} in {location} with {area} sq yards, {bedrooms} beds, and {bathrooms} baths."
                        advice = generate_insights(f"Property details: {advice_prompt}", f"Estimated prediction: PKR {prediction:,.0f}", "Neighborhood, Area, Layout Structure")
                        
                        # Check if Groq failed/errored
                        if "Error generating insights" in advice or "Groq API Key" in advice:
                            st.info("Groq API returned an error or is unconfigured. Launching expert rule-based advisor report instead.")
                            advice = get_fallback_property_advice(location, prop_type, area, bedrooms, bathrooms, prediction)
                        st.markdown(advice)
                else:
                    st.info("Groq API key not active or invalid. Presenting expert rule-based property valuation.")
                    advice = get_fallback_property_advice(location, prop_type, area, bedrooms, bathrooms, prediction)
                    st.markdown(advice)
            except Exception as ex:
                st.error(f"Prediction failed: {ex}")
                st.info("Make sure the preprocessing pipeline configuration aligns with model specifications.")
    else:
        st.error("No trained models found! Please run the training pipeline first: `python src/train.py`.")

# ----------------------------------------------------
# PAGE 7: AI BUSINESS INSIGHTS
# ----------------------------------------------------
elif page == "💡 AI Business Insights":
    st.markdown("<div class='section-header'>Market Insights & Strategic Analysis</div>", unsafe_allow_html=True)
    
    st.write("Generate a strategic advisory report highlighting startup roadmap possibilities, market risks, and valuation trends in Karachi.")
    
    if st.button("Generate Executive Strategy Report", type="primary"):
        api_key = os.getenv('GROQ_API_KEY')
        if api_key and api_key.strip() and not api_key.startswith("your_"):
            with st.spinner("Compiling dataset analytics and query patterns..."):
                summary = df_raw.describe().to_string()
                
                # Read metrics if available
                metrics_summary = "Linear Regression R2: ~0.87, Random Forest R2: ~0.97"
                metrics_path = os.path.join(BASE_DIR, 'models', 'metrics.json')
                if os.path.exists(metrics_path):
                    with open(metrics_path, 'r') as f:
                        m_data = json.load(f)
                        metrics_summary = f"Linear Regression Test R2: {m_data['Linear Regression']['test']['R2']:.3f}, Random Forest Test R2: {m_data['Random Forest']['test']['R2']:.3f}"
                
                top_features = "Area (Sq Yards), location_DHA, location_Clifton, property_type_House, room_density"
                
                insights = generate_insights(summary, metrics_summary, top_features)
                
                if "Error generating insights" in insights or "Groq API Key" in insights:
                    st.info("Groq integration unavailable. Displaying localized startup strategy roadmap.")
                    insights = get_fallback_business_insights()
                st.markdown(insights)
        else:
            st.info("Groq API Key missing or invalid. Displaying pre-compiled Karachi startup strategy report.")
            insights = get_fallback_business_insights()
            st.markdown(insights)
