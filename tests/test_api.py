from unittest.mock import patch
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from src.api.main import app


def test_invalid_age_is_rejected():
    df = pd.read_csv("tests/data/credit_sample.csv")
    payload = df.drop(columns=["ID", "default.payment.next.month"]).iloc[0].to_dict()
    payload["AGE"] = -5

    with patch("src.api.main.joblib.load") as load_model:
        with TestClient(app) as client:
            response = client.post("/predict", json=payload)

        assert response.status_code == 422
        errors = response.json()["detail"]

        assert any(
            error["loc"] == ["body", "AGE"] and error["type"] == "greater_than_equal"
            for error in errors
        )

        load_model.return_value.predict.assert_not_called()
        load_model.return_value.predict_proba.assert_not_called()


def test_valid_request_returns_prediction():
    df = pd.read_csv("tests/data/credit_sample.csv")
    payload = df.drop(columns=["ID", "default.payment.next.month"]).iloc[0].to_dict()

    with patch("src.api.main.joblib.load") as load_model:
        model = load_model.return_value
        model.predict.return_value = np.array([0])
        model.predict_proba.return_value = np.array([[0.75, 0.25]])

        with TestClient(app) as client:
            response = client.post("/predict", json=payload)

        assert response.status_code == 200
        assert response.json() == {
            "predicted_class": 0,
            "default_probability": 0.25,
        }
        model.predict.assert_called_once()
        model.predict_proba.assert_called_once()
