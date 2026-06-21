import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

def explain_model():
    import os
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Load the Random Forest model (pipeline)
    model = joblib.load(os.path.join(BASE_DIR, 'models', 'random_forest.pkl'))
    
    # Extract feature names from preprocessor
    preprocessor = model.named_steps['preprocessor']
    
    # Get categorical feature names after one-hot encoding
    cat_features = preprocessor.transformers_[1][2]
    cat_encoder = preprocessor.transformers_[1][1].named_steps['onehot']
    cat_feature_names = cat_encoder.get_feature_names_out(cat_features).tolist()
    
    # Get numerical feature names
    num_feature_names = preprocessor.transformers_[0][2]
    
    all_features = num_feature_names + cat_feature_names
    
    # Get importances
    importances = model.named_steps['regressor'].feature_importances_
    
    feature_importance_df = pd.DataFrame({
        'Feature': all_features,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    
    # Save to CSV
    csv_path = os.path.join(BASE_DIR, 'models', 'feature_importances.csv')
    feature_importance_df.to_csv(csv_path, index=False)
    print(f"Feature importances saved to {csv_path}")
    
    # Plot top 10
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(10))
    plt.title('Top 10 Feature Importances (Random Forest)')
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, 'notebooks', 'feature_importance.png'))
    plt.close()
    
    print("Feature importance plot saved to notebooks/feature_importance.png")
    return feature_importance_df

if __name__ == "__main__":
    explain_model()
