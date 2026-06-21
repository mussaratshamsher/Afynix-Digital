import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def engineer_features(df):
    """
    Creates new features and returns a dataframe with engineered features.
    """
    # Create new features
    df['price_per_sqft'] = df['price_pkr'] / (df['area_sq_yards'] * 9)
    df['room_density'] = (df['bedrooms'] + df['bathrooms']) / df['area_sq_yards']
    
    return df

def get_preprocessing_pipeline(cat_features, num_features):
    """
    Returns a scikit-learn ColumnTransformer for preprocessing.
    """
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, num_features),
            ('cat', categorical_transformer, cat_features)
        ])
    
    return preprocessor

if __name__ == "__main__":
    import os
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    df = pd.read_csv(os.path.join(BASE_DIR, 'data', 'cleaned_housing.csv'))
    df = engineer_features(df)
    df.to_csv(os.path.join(BASE_DIR, 'data', 'engineered_housing.csv'), index=False)
    print(f"Feature engineering complete. Columns: {df.columns.tolist()}")
