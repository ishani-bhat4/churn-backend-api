from fastapi import FastAPI
from pydantic import BaseModel
from joblib import load
import pandas as pd
import numpy as np

app = FastAPI(title="Customer Churn Predictor")

pipe = load("churn_pipeline.joblib")

DEFAULT_THRESHOLD = 0.3


class ChurnRequest(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


def explain_prediction(pipe, X_row: pd.DataFrame, top_n: int = 5):
    """
    Return top positive and top negative feature contributions for ONE row.
    Positive contribution => pushes towards churn (class 1)
    Negative contribution => pushes away from churn (retention)
    """
    preprocess = pipe.named_steps["preprocess"]
    model = pipe.named_steps["model"]

    Xt = preprocess.transform(X_row)
    if hasattr(Xt, "toarray"):
        Xt = Xt.toarray()

    x = Xt[0]                       # transformed feature vector
    coefs = model.coef_[0]          # logistic coefficients

    contributions = x * coefs

    # Names of transformed features (includes one-hot output names)
    feature_names = preprocess.get_feature_names_out()

    exp = pd.DataFrame({
        "feature": feature_names,
        "contribution": contributions
    })

    # Keep only non-zero-ish contributions (optional; makes output cleaner)
    exp = exp[exp["contribution"].abs() > 1e-12]

    top_risk = (
        exp.sort_values("contribution", ascending=False)
        .head(top_n)["feature"]
        .tolist()
    )

    top_protect = (
        exp.sort_values("contribution", ascending=True)
        .head(top_n)["feature"]
        .tolist()
    )

    return top_risk, top_protect


@app.post("/predict")
def predict(req: ChurnRequest, threshold: float = DEFAULT_THRESHOLD):
    X = pd.DataFrame([req.dict()])

    prob = float(pipe.predict_proba(X)[0, 1])
    risk = "High" if prob >= threshold else "Low"

    top_risk, top_protect = explain_prediction(pipe, X, top_n=5)

    return {
        "churn_probability": round(prob, 3),
        "risk": risk,
        "threshold_used": float(threshold),
        "top_risk_factors": top_risk,
        "top_protective_factors": top_protect
    }
