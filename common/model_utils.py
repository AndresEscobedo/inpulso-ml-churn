"""Helpers for loading churn models and running inference."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import joblib
import pandas as pd

from common.churn_config import (
    BINARY_MAPPINGS,
    CATEGORICAL_DUMMIES,
    CHURN_FEATURES,
    MODEL_REGISTRY_PATH,
)


def _normalize_key(key: str) -> str:
    return (
        key.strip()
        .lower()
        .replace(" ", "")
        .replace("_", "")
        .replace("/", "")
    )


def normalize_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    normalized: Dict[str, Any] = {}
    for key, value in payload.items():
        normalized[_normalize_key(str(key))] = value
    return normalized


def prepare_features(payload: Dict[str, Any]) -> pd.DataFrame:
    normalized = normalize_payload(payload)
    frame = pd.DataFrame([normalized])
    for column in CHURN_FEATURES:
        if column not in frame.columns:
            frame[column] = pd.NA
    # Normalize categorical
    for col in CATEGORICAL_DUMMIES:
        if col in frame.columns:
            frame[col] = frame[col].astype(str).str.strip().str.lower()
    # Map binary fields
    bin_map = {"yes": 1, "no": 0, True: 1, False: 0, 1: 1, 0: 0}
    for col in BINARY_MAPPINGS:
        if col in frame.columns:
            frame[col] = frame[col].map(bin_map)
    # Coerce numerics where possible
    for col in frame.columns:
        if col not in CATEGORICAL_DUMMIES:
            frame[col] = pd.to_numeric(frame[col], errors="ignore")
    return frame[CHURN_FEATURES]


def load_artifact(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Model artifact not found at {path}")
    return joblib.load(path)


def registry_entry(model_key: str) -> Dict[str, Any]:
    if not MODEL_REGISTRY_PATH.exists():
        return {}
    try:
        with MODEL_REGISTRY_PATH.open("r", encoding="utf-8") as registry_file:
            data = json.load(registry_file)
            return data.get(model_key, {})
    except json.JSONDecodeError:
        return {}
