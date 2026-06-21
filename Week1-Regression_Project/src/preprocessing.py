import pandas as pd
import numpy as np

def clean_data(df):
    """
    Cleans the dataframe: handles missing values, duplicates, and outliers.
    """
    # 1. Remove duplicates
    df = df.drop_duplicates()
    
    # 2. Handle missing values
    # Numerical: Median imputation
    num_cols = df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())
        
    # Categorical: Mode imputation
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])
        
    # 3. Handle outliers using IQR Method
    for col in ['area_sq_yards', 'price_pkr']:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
        
    return df

if __name__ == "__main__":
    import os
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_data = pd.read_csv(os.path.join(BASE_DIR, 'data', 'housing.csv'))
    cleaned_data = clean_data(raw_data)
    cleaned_data.to_csv(os.path.join(BASE_DIR, 'data', 'cleaned_housing.csv'), index=False)
    print(f"Data cleaned. Rows before: {len(raw_data)}, Rows after: {len(cleaned_data)}")
