---
title: Customer Churn Prediction API
emoji: 📊
colorFrom: blue
colorTo: green
sdk: docker
app_file: Dockerfile
python_version: "3.10"
pinned: false
---

# 📊 Customer Churn Prediction API

FastAPI-based machine learning service for predicting customer churn. Deploys as an API (not dashboard).

## API Endpoints

| Endpoint | Method | Description |
|---------|--------|-------------|
| `/` | GET | Health check |
| `/predict` | POST | Predict single customer |
| `/predict_batch` | POST | Predict multiple customers (JSON) |
| `/predict_csv` | POST | Upload CSV file for batch prediction |

## Features

- FastAPI REST API
- CSV file upload support
- Batch prediction
- CORS enabled for cross-origin integration
- Docker deployment

## 📁 Project Structure

```text
week2-Classification_Project/
├── app/
│   ├── api.py
│   └── dashboard.py
├── data/
│   └── raw/
│       └── churn.csv
├── models/
├── outputs/
├── reports/
├── src/
├── requirements.txt
└── README.md