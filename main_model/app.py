"""FastAPI service for the main churn model."""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, HTTPException

from common.churn_config import CHURN_FEATURES, MAIN_MODEL_ARTIFACT
from common.model_utils import load_artifact, prepare_features, registry_entry
from common.schemas import ChurnRequest

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_PATH = Path(os.getenv("MODEL_ARTIFACT_PATH", str(MAIN_MODEL_ARTIFACT)))
MODEL_KEY = os.getenv("MODEL_KEY", "main")

try:
    loaded_model = load_artifact(MODEL_PATH)
    logger.info("Loaded churn model from %s", MODEL_PATH)
except FileNotFoundError as exc:
    logger.exception("Model artifact missing: %s", exc)
    raise

app = FastAPI(title="Main Churn Model API")


def _predict(payload: Dict[str, Any]) -> Dict[str, Any]:
    feature_frame = prepare_features(payload)
    probabilities = loaded_model.predict_proba(feature_frame)[0]
    churn_probability = float(probabilities[-1])
    prediction = int(loaded_model.predict(feature_frame)[0])
    return {
        "model_key": MODEL_KEY,
        "features_used": CHURN_FEATURES,
        "predicted_label": prediction,
        "churn_probability": churn_probability,
        "probability_vector": probabilities.tolist(),
    }


@app.post("/predict")
async def predict(request: ChurnRequest):
    try:
        result = _predict(request.model_dump())
        response = {"input": request.model_dump(), "prediction": result}
        logger.debug("Prediction response: %s", response)
        return response
    except Exception as exc:  # pragma: no cover - surfaced via HTTP error
        logger.exception("Prediction failed")
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_key": MODEL_KEY, "artifact": str(MODEL_PATH)}


@app.get("/metadata")
async def model_metadata():
    metadata = registry_entry(MODEL_KEY)
    return {
        "model_key": MODEL_KEY,
        "artifact_path": str(MODEL_PATH),
        "metrics": metadata.get("metrics", {}),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=5000)
