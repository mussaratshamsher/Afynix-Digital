import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from preprocessing import clean_data
from feature_engineering import engineer_features, get_preprocessing_pipeline

def train_models():
    import os
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Load and clean data
    df = pd.read_csv(os.path.join(BASE_DIR, 'data', 'housing.csv'))
    df = clean_data(df)
    df = engineer_features(df)
    
    # Define features and target
    X = df.drop(['price_pkr', 'price_per_sqft'], axis=1)
    y = df['price_pkr']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Identify categorical and numerical features
    cat_features = X.select_dtypes(include=['object']).columns.tolist()
    num_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Preprocessing pipeline
    preprocessor = get_preprocessing_pipeline(cat_features, num_features)
    
    # Model 1: Linear Regression
    lr_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ])
    lr_pipeline.fit(X_train, y_train)
    joblib.dump(lr_pipeline, os.path.join(BASE_DIR, 'models', 'linear_regression.pkl'))
    
    # Model 2: Random Forest with GridSearchCV
    rf_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(random_state=42))
    ])
    
    param_grid = {
        'regressor__n_estimators': [50, 100],
        'regressor__max_depth': [None, 10, 20],
        'regressor__min_samples_split': [2, 5]
    }
    
    grid_search = GridSearchCV(rf_pipeline, param_grid, cv=3, scoring='r2', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    best_rf_model = grid_search.best_estimator_
    joblib.dump(best_rf_model, os.path.join(BASE_DIR, 'models', 'random_forest.pkl'))
    
    print("Models trained and saved to models/ directory.")
    return X_test, y_test

if __name__ == "__main__":
    train_models()
