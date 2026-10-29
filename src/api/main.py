from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, Request

from src.api.schemas import ClientFeatures
from src.features.build_features import feature_engineering


@asynccontextmanager
async def lifespan(app: FastAPI):
    project_root = Path(__file__).resolve().parents[2]
    model_path = project_root / "models" / "trained" / "model.joblib"

    app.state.model = joblib.load(model_path)
    yield
    del app.state.model


app = FastAPI(
    title="Credit Scoring API",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(client: ClientFeatures, request: Request):
    data = pd.DataFrame([client.model_dump()])
    features = feature_engineering(data)

    model = request.app.state.model
    predicted_class = int(model.predict(features)[0])
    probability = float(model.predict_proba(features)[0, 1])

    return {
        "predicted_class": predicted_class,
        "default_probability": probability,
    }
