import io

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from app.main import app
from app.ml import analyse_labeled_dataset, read_dataset


def make_dataset(n=120):
    rng = np.random.default_rng(42)
    tenure = rng.integers(1, 72, size=n)
    monthly = rng.normal(70, 18, size=n).clip(20, 130)
    contract = rng.choice(["Month-to-month", "One year", "Two year"], size=n, p=[0.55, 0.25, 0.20])
    logit = 1.2 - 0.035 * tenure + 0.018 * (monthly - 60) + (contract == "Month-to-month") * 0.8
    prob = 1 / (1 + np.exp(-logit))
    churn = np.where(rng.random(n) < prob, "Yes", "No")
    return pd.DataFrame({
        "customerID": [f"C{i:05}" for i in range(n)],
        "tenure": tenure,
        "MonthlyCharges": monthly.round(2),
        "Contract": contract,
        "Churn": churn,
    })


def test_analysis_returns_oof_metrics_and_probabilities():
    result, artifact = analyse_labeled_dataset(make_dataset(), save_model=True)
    assert result["status"] == "ok"
    assert result["cross_validation"]["probabilities"] == "out-of-fold"
    assert 0.0 <= result["evaluation"]["roc_auc"] <= 1.0
    assert result["prediction_count_returned"] == 120
    assert artifact is not None
    assert "customerID" in result["dataset"]["dropped_feature_columns"]
    assert result["predictions"][0]["risk_level"] in {"Low", "Medium", "High"}


def test_csv_reader_and_target_detection():
    csv_bytes = make_dataset(30).to_csv(index=False).encode()
    df = read_dataset(csv_bytes, "test.csv")
    result, _ = analyse_labeled_dataset(df)
    assert result["dataset"]["target_column"] == "Churn"


def test_api_health():
    client = TestClient(app)
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
