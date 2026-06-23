from __future__ import annotations

import os
import numpy as np
import pandas as pd

def generate_dummy_data():
    np.random.seed(42)
    n_samples = 1000

    # Generate columns
    customer_ids = [f"{np.random.randint(1000, 9999)}-{np.random.choice(list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'), 5)}" for _ in range(n_samples)]
    genders = np.random.choice(["Male", "Female"], size=n_samples)
    senior_citizens = np.random.choice([0, 1], size=n_samples, p=[0.85, 0.15])
    partners = np.random.choice(["Yes", "No"], size=n_samples)
    dependents = np.random.choice(["Yes", "No"], size=n_samples, p=[0.7, 0.3])
    tenure = np.random.randint(0, 73, size=n_samples)  # 0 indicates new customer
    phone_service = np.random.choice(["Yes", "No"], size=n_samples, p=[0.9, 0.1])

    multiple_lines = []
    for ps in phone_service:
        if ps == "No":
            multiple_lines.append("No phone service")
        else:
            multiple_lines.append(np.random.choice(["Yes", "No"], p=[0.4, 0.6]))

    internet_service = np.random.choice(["Fiber optic", "DSL", "No"], size=n_samples, p=[0.4, 0.4, 0.2])

    def get_internet_addon(is_service):
        if is_service == "No":
            return "No internet service"
        return np.random.choice(["Yes", "No"], p=[0.35, 0.65])

    online_security = [get_internet_addon(i) for i in internet_service]
    online_backup = [get_internet_addon(i) for i in internet_service]
    device_protection = [get_internet_addon(i) for i in internet_service]
    tech_support = [get_internet_addon(i) for i in internet_service]
    streaming_tv = [get_internet_addon(i) for i in internet_service]
    streaming_movies = [get_internet_addon(i) for i in internet_service]

    contract = np.random.choice(["Month-to-month", "One year", "Two year"], size=n_samples, p=[0.55, 0.20, 0.25])
    paperless_billing = np.random.choice(["Yes", "No"], size=n_samples, p=[0.6, 0.4])
    payment_method = np.random.choice(
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        size=n_samples,
        p=[0.35, 0.25, 0.2, 0.2]
    )

    # Charges: monthly charges depend on services
    monthly_charges = []
    for i in range(n_samples):
        base = 20.0
        if internet_service[i] == "Fiber optic":
            base += 50.0
        elif internet_service[i] == "DSL":
            base += 30.0
        if phone_service[i] == "Yes":
            base += 10.0
        if online_security[i] == "Yes":
            base += 5.0
        if online_backup[i] == "Yes":
            base += 5.0
        if device_protection[i] == "Yes":
            base += 5.0
        if tech_support[i] == "Yes":
            base += 5.0
        if streaming_tv[i] == "Yes":
            base += 8.0
        if streaming_movies[i] == "Yes":
            base += 8.0
        # Add minor noise
        base += np.random.uniform(-3.0, 3.0)
        monthly_charges.append(round(base, 2))

    total_charges = []
    for i in range(n_samples):
        if tenure[i] == 0:
            total_charges.append(" ") # Simulate empty string/space for 0 tenure
        else:
            tc = monthly_charges[i] * tenure[i] + np.random.uniform(-5.0, 5.0)
            total_charges.append(str(round(max(tc, monthly_charges[i]), 2)))

    # Generate Churn target
    churn_probs = []
    for i in range(n_samples):
        p = 0.05
        if contract[i] == "Month-to-month":
            p += 0.35
        if tenure[i] < 12:
            p += 0.25
        elif tenure[i] < 24:
            p += 0.10
        if payment_method[i] == "Electronic check":
            p += 0.15
        if internet_service[i] == "Fiber optic":
            p += 0.10
        p = min(max(p, 0.01), 0.95)
        churn_probs.append(p)

    churn = [("Yes" if np.random.rand() < p else "No") for p in churn_probs]

    df = pd.DataFrame({
        "customerID": customer_ids,
        "gender": genders,
        "SeniorCitizen": senior_citizens,
        "Partner": partners,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
        "Churn": churn
    })

    # Make directory and save
    raw_dir = os.path.join(os.path.dirname(__file__), "data", "raw")
    os.makedirs(raw_dir, exist_ok=True)
    df.to_csv(os.path.join(raw_dir, "churn.csv"), index=False)
    print(f"Dummy dataset created with {len(df)} rows at {os.path.join(raw_dir, 'churn.csv')}")

if __name__ == "__main__":
    generate_dummy_data()
