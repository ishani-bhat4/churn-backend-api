# 🚀 Customer Churn Prediction API

This repository contains the **FastAPI backend** for the Customer Churn Risk Prediction system.

It serves a trained machine learning pipeline and exposes an inference endpoint used by the Streamlit frontend.

---

## 🌐 Live API

- **Base URL**:  
  https://churn-backend-api-nxc0.onrender.com

- **Interactive Docs (Swagger)**:  
  https://churn-backend-api-nxc0.onrender.com/docs

---

## 📡 API Endpoints

### `POST /predict`

Predicts churn probability and returns model explanations.

**Query Params**
- `threshold` (float): Decision threshold (default = 0.3)

**Request Body (JSON)**
```json
{
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "Yes",
  "tenure": 15,
  "PhoneService": "No",
  "MultipleLines": "No phone service",
  "InternetService": "No",
  "OnlineSecurity": "No internet service",
  "OnlineBackup": "No internet service",
  "DeviceProtection": "No internet service",
  "TechSupport": "No internet service",
  "StreamingTV": "No internet service",
  "StreamingMovies": "Yes",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Bank transfer (automatic)",
  "MonthlyCharges": 70.0,
  "TotalCharges": 1000.0
}
**Response**
{
  "churn_probability": 0.067,
  "risk": "Low",
  "threshold_used": 0.3,
  "top_risk_factors": [...],
  "top_protective_factors": [...]
}

