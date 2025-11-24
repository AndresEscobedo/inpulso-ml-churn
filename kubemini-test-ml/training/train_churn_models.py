"""Train churn models once and persist artifacts for serving."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from common.churn_config import (
    ARTIFACT_DIR,
    CANARY_MODEL_ARTIFACT,
    CATEGORICAL_FEATURES,
    CHURN_FEATURES,
    DATA_PATH,
    MAIN_MODEL_ARTIFACT,
    MODEL_REGISTRY_PATH,
    NUMERIC_FEATURES,
    RANDOM_STATE,
    TARGET_COLUMN,
)


def _load_dataset() -> pd.DataFrame:
    dataset = pd.read_csv(DATA_PATH)
    dataset.columns = (
        dataset.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace("/", "_", regex=False)
    )
    dataset = dataset.replace({"": pd.NA, "na": pd.NA, "NA": pd.NA})
    dataset = dataset.drop_duplicates()
    dataset[TARGET_COLUMN] = (
        dataset[TARGET_COLUMN]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({"yes": 1, "no": 0})
    )
    dataset = dataset.dropna(subset=[TARGET_COLUMN])
    for col in NUMERIC_FEATURES:
        dataset[col] = pd.to_numeric(dataset[col], errors="coerce")
    return dataset


def _build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )


def _train_pipeline(classifier) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocess", _build_preprocessor()),
            ("classifier", classifier),
        ]
    )


def _evaluate(model: Pipeline, x_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
    predictions = model.predict(x_test)
    probabilities = model.predict_proba(x_test)[:, 1]
    return {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "f1": float(f1_score(y_test, predictions)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
    }


def train() -> Tuple[Dict[str, float], Dict[str, float]]:
    data = _load_dataset()
    feature_frame = data[CHURN_FEATURES]
    target = data[TARGET_COLUMN]

    x_train, x_test, y_train, y_test = train_test_split(
        feature_frame,
        target,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=target,
    )

    main_pipeline = _train_pipeline(
        RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    )

    canary_pipeline = _train_pipeline(
        GradientBoostingClassifier(
            random_state=RANDOM_STATE,
            learning_rate=0.08,
            n_estimators=200,
            max_depth=3,
        )
    )

    main_pipeline.fit(x_train, y_train)
    canary_pipeline.fit(x_train, y_train)

    main_metrics = _evaluate(main_pipeline, x_test, y_test)
    canary_metrics = _evaluate(canary_pipeline, x_test, y_test)

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(main_pipeline, MAIN_MODEL_ARTIFACT)
    joblib.dump(canary_pipeline, CANARY_MODEL_ARTIFACT)

    with MODEL_REGISTRY_PATH.open("w", encoding="utf-8") as registry_file:
        json.dump(
            {
                "main": {"metrics": main_metrics},
                "canary": {"metrics": canary_metrics},
            },
            registry_file,
            indent=2,
        )

    return main_metrics, canary_metrics


if __name__ == "__main__":
    main_results, canary_results = train()
    print("Main model metrics:", json.dumps(main_results, indent=2))
    print("Canary model metrics:", json.dumps(canary_results, indent=2))
