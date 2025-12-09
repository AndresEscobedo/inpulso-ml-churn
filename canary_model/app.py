"""FastAPI service for the canary churn model."""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import shap

from fastapi import FastAPI, HTTPException

from common.churn_config import CHURN_FEATURES, CANARY_MODEL_ARTIFACT, ARTIFACT_DIR
from common.model_utils import load_artifact, prepare_features, registry_entry
from common.schemas import ChurnRequest

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_PATH = Path(os.getenv("MODEL_ARTIFACT_PATH", str(CANARY_MODEL_ARTIFACT)))
MODEL_KEY = os.getenv("MODEL_KEY", "canary")
BACKGROUND_DATA_PATH = ARTIFACT_DIR / "background_data.joblib"

try:
    loaded_model = load_artifact(MODEL_PATH)
    logger.info("Loaded canary model from %s", MODEL_PATH)
except FileNotFoundError as exc:
    logger.exception("Model artifact missing: %s", exc)
    raise

app = FastAPI(title="Canary Churn Model API")


def _get_explainer():
    try:
        background = load_artifact(BACKGROUND_DATA_PATH)
        logger.info("Loaded background data for SHAP from %s", BACKGROUND_DATA_PATH)
    except Exception:
        logger.warning("Background data not found, falling back to zeros (may cause issues with categoricals)")
        background = pd.DataFrame([np.zeros(len(CHURN_FEATURES))], columns=CHURN_FEATURES)

    return shap.KernelExplainer(
        lambda data: loaded_model.predict_proba(pd.DataFrame(data, columns=CHURN_FEATURES))[:, 1],
        background,
    )


_explainer = _get_explainer()


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


def _explain(payload: Dict[str, Any]) -> Dict[str, Any]:
    feature_frame = prepare_features(payload)
    shap_values = _explainer.shap_values(feature_frame)
    contrib = shap_values[0].tolist()
    base_value = float(_explainer.expected_value)
    return {
        "model_key": MODEL_KEY,
        "features": CHURN_FEATURES,
        "base_value": base_value,
        "contributions": contrib,
    }


@app.post("/predict")
async def predict(request: ChurnRequest):
    try:
        result = _predict(request.model_dump())
        return {"input": request.model_dump(), "prediction": result}
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/explain")
async def explain(request: ChurnRequest):
    try:
        result = _explain(request.model_dump())
        return {"input": request.model_dump(), "explanation": result}
    except Exception as exc:
        logger.exception("Explain failed")
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

    uvicorn.run(app, host="0.0.0.0", port=5001)
