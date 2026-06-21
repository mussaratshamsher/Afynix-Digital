import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from preprocessing import clean_data
from feature_engineering import engineer_features
from sklearn.model_selection import train_test_split

def evaluate_models():
    import os
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Load and clean data
    df = pd.read_csv(os.path.join(BASE_DIR, 'data', 'housing.csv'))
    df = clean_data(df)
    df = engineer_features(df)
    
    X = df.drop(['price_pkr', 'price_per_sqft'], axis=1)
    y = df['price_pkr']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    models = {
        'Linear Regression': joblib.load(os.path.join(BASE_DIR, 'models', 'linear_regression.pkl')),
        'Random Forest': joblib.load(os.path.join(BASE_DIR, 'models', 'random_forest.pkl'))
    }
    
    results = []
    
    for name, model in models.items():
        y_pred = model.predict(X_test)
        
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        
        results.append({
            'Model': name,
            'R2': r2,
            'MAE': mae,
            'RMSE': rmse
        })
        
        # Plotting Actual vs Predicted
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        plt.title(f'{name}: Actual vs Predicted')
        plt.xlabel('Actual Price (PKR)')
        plt.ylabel('Predicted Price (PKR)')
        plt.tight_layout()
        img_path = os.path.join(BASE_DIR, 'notebooks', f'{name.lower().replace(" ", "_")}_actual_vs_pred.png')
        plt.savefig(img_path)
        plt.close()

    results_df = pd.DataFrame(results)
    print("\nModel Comparison Table:")
    print(results_df)
    return results_df

if __name__ == "__main__":
    evaluate_models()
