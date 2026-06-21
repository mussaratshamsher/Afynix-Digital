import pandas as pd
import numpy as np
import os

def generate_karachi_housing_data(n_samples=1200):
    np.random.seed(42)
    
    locations = {
        'DHA': {'base_price': 50000000, 'price_per_sq_yard': 250000},
        'Clifton': {'base_price': 45000000, 'price_per_sq_yard': 220000},
        'Bahria Town': {'base_price': 15000000, 'price_per_sq_yard': 80000},
        'Gulshan-e-Iqbal': {'base_price': 25000000, 'price_per_sq_yard': 120000},
        'North Nazimabad': {'base_price': 20000000, 'price_per_sq_yard': 100000},
        'Malir': {'base_price': 10000000, 'price_per_sq_yard': 50000},
        'Korangi': {'base_price': 8000000, 'price_per_sq_yard': 40000},
        'Federal B Area': {'base_price': 18000000, 'price_per_sq_yard': 90000}
    }
    
    data = []
    
    for _ in range(n_samples):
        loc = np.random.choice(list(locations.keys()))
        area = np.random.randint(80, 1001)  # Square yards
        
        # Base price calculation
        base = locations[loc]['base_price']
        var = locations[loc]['price_per_sq_yard']
        
        # Pricing logic
        price = base + (area * var) + np.random.normal(0, base * 0.1)
        
        # Features
        bedrooms = max(1, int(area / 100) + np.random.randint(0, 3))
        bathrooms = max(1, bedrooms - np.random.randint(0, 2))
        
        prop_type = np.random.choice(['House', 'Flat', 'Penthouse'], p=[0.6, 0.35, 0.05])
        
        # Adjust price for type
        if prop_type == 'Flat': price *= 0.7
        if prop_type == 'Penthouse': price *= 1.5
        
        data.append({
            'location': loc,
            'area_sq_yards': area,
            'bedrooms': bedrooms,
            'bathrooms': bathrooms,
            'property_type': prop_type,
            'price_pkr': round(price, -3)
        })
    
    df = pd.DataFrame(data)
    
    # Introduce some missing values (5%)
    for col in ['bedrooms', 'bathrooms']:
        df.loc[df.sample(frac=0.05).index, col] = np.nan
        
    # Introduce some duplicates
    df = pd.concat([df, df.sample(n=50)], ignore_index=True)
    
    # Save to CSV
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/housing.csv', index=False)
    print(f"Dataset generated: data/housing.csv with {len(df)} records.")

if __name__ == "__main__":
    generate_karachi_housing_data()
