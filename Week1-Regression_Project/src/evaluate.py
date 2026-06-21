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
    
    results = {}
    
    for name, model in models.items():
        # Predict on train
        y_train_pred = model.predict(X_train)
        r2_train = r2_score(y_train, y_train_pred)
        mae_train = mean_absolute_error(y_train, y_train_pred)
        rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
        
        # Predict on test
        y_test_pred = model.predict(X_test)
        r2_test = r2_score(y_test, y_test_pred)
        mae_test = mean_absolute_error(y_test, y_test_pred)
        rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
        
        # Overfitting check
        r2_gap = r2_train - r2_test
        if r2_gap > 0.05:
            overfitting_status = "Potential Overfitting (High Gap)"
            overfitting_desc = f"The model performs significantly better on training data (R²: {r2_train:.3f}) than on testing data (R²: {r2_test:.3f}). Gap is {r2_gap:.3f}."
        elif r2_train < 0.60 and r2_test < 0.60:
            overfitting_status = "Potential Underfitting (Low Score)"
            overfitting_desc = "The model has low scores on both training and testing sets, indicating it might be underfitting the dataset."
        else:
            overfitting_status = "Healthy / Well Generalized"
            overfitting_desc = "The training and testing performances are close, suggesting the model generalizes well to unseen data."

        results[name] = {
            'train': {
                'R2': float(r2_train),
                'MAE': float(mae_train),
                'RMSE': float(rmse_train)
            },
            'test': {
                'R2': float(r2_test),
                'MAE': float(mae_test),
                'RMSE': float(rmse_test)
            },
            'overfitting': {
                'status': overfitting_status,
                'description': overfitting_desc,
                'gap': float(r2_gap)
            }
        }
        
        # Plotting 1: Actual vs Predicted
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_test_pred, alpha=0.5, color='royalblue')
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.title(f'{name}: Actual vs Predicted')
        plt.xlabel('Actual Price (PKR)')
        plt.ylabel('Predicted Price (PKR)')
        plt.tight_layout()
        img_path = os.path.join(BASE_DIR, 'notebooks', f'{name.lower().replace(" ", "_")}_actual_vs_pred.png')
        plt.savefig(img_path)
        plt.close()

        # Plotting 2: Residuals Plot
        residuals = y_test - y_test_pred
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test_pred, residuals, alpha=0.5, color='darkorange')
        plt.axhline(y=0, color='r', linestyle='--', lw=2)
        plt.title(f'{name}: Residuals vs Fitted')
        plt.xlabel('Fitted Values (Predicted Price)')
        plt.ylabel('Residuals (Actual - Predicted)')
        plt.tight_layout()
        res_img_path = os.path.join(BASE_DIR, 'notebooks', f'{name.lower().replace(" ", "_")}_residuals.png')
        plt.savefig(res_img_path)
        plt.close()

    # Save to JSON
    import json
    metrics_path = os.path.join(BASE_DIR, 'models', 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Metrics saved to {metrics_path}")

    # Format results for console comparison
    comparison_records = []
    for model_name, metrics in results.items():
        comparison_records.append({
            'Model': model_name,
            'Train R2': metrics['train']['R2'],
            'Test R2': metrics['test']['R2'],
            'Train MAE': metrics['train']['MAE'],
            'Test MAE': metrics['test']['MAE'],
            'Overfitting Status': metrics['overfitting']['status']
        })
    results_df = pd.DataFrame(comparison_records)
    print("\nModel Comparison Table:")
    print(results_df.to_string(index=False))
    return results_df


if __name__ == "__main__":
    evaluate_models()
