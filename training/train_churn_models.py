"""Train churn models once and persist artifacts for serving."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple, List

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
    BINARY_MAPPINGS,
    CANARY_MODEL_ARTIFACT,
    CATEGORICAL_DUMMIES,
    CHURN_FEATURES,
    DATA_PATH,
    MAIN_MODEL_ARTIFACT,
    MODEL_REGISTRY_PATH,
    NUMERIC_FEATURES,
    RANDOM_STATE,
    TARGET_COLUMN,
)


def _normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.rename(columns=lambda c: c.strip().lower().replace(" ", "_").replace("/", "_"))


def _load_dataset() -> pd.DataFrame:
    dataset = pd.read_csv(DATA_PATH)
    dataset = _normalize_columns(dataset)
    dataset = dataset.dropna()
    dataset = dataset.drop(columns=["employeenumber", "over18", "standardhours"], errors="ignore")

    # Normalize categorical strings
    for col in CATEGORICAL_DUMMIES + BINARY_MAPPINGS:
        if col in dataset.columns:
            dataset[col] = dataset[col].astype(str).str.strip().str.lower()

    dict_overtime = {"yes": 1, "no": 0}
    dict_disobedience = {"yes": 1, "no": 0}
    dict_attrition = {"yes": 1, "no": 0}

    dataset["overtime"] = dataset["overtime"].map(dict_overtime)
    dataset["disobediencerules"] = dataset["disobediencerules"].map(dict_disobedience)
    dataset[TARGET_COLUMN] = dataset[TARGET_COLUMN].map(dict_attrition)

    dataset = dataset.dropna(subset=[TARGET_COLUMN])

    for col in NUMERIC_FEATURES:
        if col in dataset.columns:
            dataset[col] = pd.to_numeric(dataset[col], errors="coerce")

    dataset = dataset.dropna()
    return dataset


def _build_preprocessor(dummy_columns: List[str]) -> ColumnTransformer:
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop="first"),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("cat", categorical_pipeline, dummy_columns),
        ],
        remainder="passthrough",
        sparse_threshold=0.0,
    )


def _train_pipeline(classifier, dummy_columns: List[str]) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocess", _build_preprocessor(dummy_columns)),
            ("scaler", StandardScaler(with_mean=False)),
            ("classifier", classifier),
        ]
    )


def _evaluate(model: Pipeline, x: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
    predictions = model.predict(x)
    probabilities = model.predict_proba(x)[:, 1]
    return {
        "accuracy": float(accuracy_score(y, predictions)),
        "f1": float(f1_score(y, predictions)),
        "roc_auc": float(roc_auc_score(y, probabilities)),
    }


def train() -> Tuple[Dict[str, float], Dict[str, float]]:
    data = _load_dataset()
    target = data[TARGET_COLUMN]
    feature_frame = data.drop(columns=[TARGET_COLUMN])

    dummy_columns = [col for col in CATEGORICAL_DUMMIES if col in feature_frame.columns]
    passthrough_columns = [col for col in feature_frame.columns if col not in dummy_columns]

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
        ),
        dummy_columns,
    )

    canary_pipeline = _train_pipeline(
        GradientBoostingClassifier(
            random_state=RANDOM_STATE,
            learning_rate=0.08,
            n_estimators=200,
            max_depth=3,
        ),
        dummy_columns,
    )

    main_pipeline.fit(x_train, y_train)
    canary_pipeline.fit(x_train, y_train)

    main_metrics_test = _evaluate(main_pipeline, x_test, y_test)
    canary_metrics_test = _evaluate(canary_pipeline, x_test, y_test)
    main_metrics_train = _evaluate(main_pipeline, x_train, y_train)
    canary_metrics_train = _evaluate(canary_pipeline, x_train, y_train)

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(main_pipeline, MAIN_MODEL_ARTIFACT)
    joblib.dump(canary_pipeline, CANARY_MODEL_ARTIFACT)

    with MODEL_REGISTRY_PATH.open("w", encoding="utf-8") as registry_file:
        json.dump(
            {
                "main": {
                    "train": main_metrics_train,
                    "test": main_metrics_test,
                },
                "canary": {
                    "train": canary_metrics_train,
                    "test": canary_metrics_test,
                },
                "meta": {
                    "dummy_columns": dummy_columns,
                    "passthrough_columns": passthrough_columns,
                },
            },
            registry_file,
            indent=2,
        )

    return main_metrics_test, canary_metrics_test


if __name__ == "__main__":
    main_results, canary_results = train()
    print("Main model metrics (test):", json.dumps(main_results, indent=2))
    print("Canary model metrics (test):", json.dumps(canary_results, indent=2))
