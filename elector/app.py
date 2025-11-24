"""Traffic splitting service for churn models."""
from __future__ import annotations

import logging
import os
import random
from typing import Any, Dict, List

import aiohttp
from fastapi import FastAPI, HTTPException

from common.schemas import ChurnRequest

FORMAT = "[%(asctime)-15s][%(levelname)-8s]%(message)s"
logging.basicConfig(format=FORMAT, level=logging.INFO)
logger = logging.getLogger(__name__)

CANARY_WEIGHT = float(os.getenv("CANARY_WEIGHT", "20"))  # percent of traffic
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT_SECONDS", "5"))
MAIN_MODEL_URL = os.getenv("MAIN_MODEL_URL", "http://model:5000")
CANARY_MODEL_URL = os.getenv("CANARY_MODEL_URL", "http://canary:5001")

MODEL_ENDPOINTS = {
    "main": f"{MAIN_MODEL_URL.rstrip('/')}/predict",
    "canary": f"{CANARY_MODEL_URL.rstrip('/')}/predict",
}

app = FastAPI(title="Churn Elector API")


def _routing_order() -> List[str]:
    primary = "canary" if random.uniform(0, 100) < CANARY_WEIGHT else "main"
    secondary = "main" if primary == "canary" else "canary"
    return [primary, secondary]


async def _call_model(session: aiohttp.ClientSession, model_key: str, payload: Dict[str, Any]):
    url = MODEL_ENDPOINTS[model_key]
    logger.info("Calling %s", url)
    async with session.post(url, json=payload, timeout=REQUEST_TIMEOUT) as response:
        text = await response.text()
        if response.status >= 400:
            logger.warning("%s returned %s: %s", url, response.status, text)
            raise HTTPException(status_code=response.status, detail=text)
        return await response.json()


async def route_prediction(payload: Dict[str, Any]):
    async with aiohttp.ClientSession() as session:
        for model_key in _routing_order():
            try:
                result = await _call_model(session, model_key, payload)
                return {"routed_to": model_key, "response": result}
            except Exception as exc:  # pragma: no cover - logged and retried
                logger.warning("Model %s failed: %s", model_key, exc)
        raise HTTPException(status_code=503, detail="No models available")


@app.post("/predict")
async def predict(request: ChurnRequest):
    payload = request.model_dump()
    return await route_prediction(payload)


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "canary_weight": CANARY_WEIGHT,
        "main_endpoint": MODEL_ENDPOINTS["main"],
        "canary_endpoint": MODEL_ENDPOINTS["canary"],
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=5002)
