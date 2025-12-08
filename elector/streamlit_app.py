import json
import math
import os
import random
from typing import Dict, Any, Tuple

import httpx
import pandas as pd
import plotly.express as px
import streamlit as st

from common.churn_config import MODEL_REGISTRY_PATH, CHURN_FEATURES
from common.model_utils import prepare_features

# Prefill sample (class 0-ish) from README
DEFAULT_SAMPLE = {
    "age": 37,
    "businesstravel": "Travel_Rarely",
    "dailyrate": 1020,
    "department": "Research & Development",
    "distancefromhome": 6,
    "education": 3,
    "educationfield": "Medical",
    "environmentsatisfaction": 4,
    "gender": "Male",
    "hourlyrate": 68,
    "jobinvolvement": 3,
    "joblevel": 2,
    "jobrole": "Research Scientist",
    "disobediencerules": "No",
    "jobsatisfaction": 4,
    "maritalstatus": "Married",
    "monthlyincome": 5500,
    "monthlyrate": 14200,
    "numcompaniesworked": 2,
    "overtime": "No",
    "percentsalaryhike": 12,
    "performancerating": 3,
    "relationshipsatisfaction": 3,
    "stockoptionlevel": 1,
    "totalworkingyears": 11,
    "trainingtimeslastyear": 3,
    "worklifebalance": 3,
    "yearsatcompany": 8,
    "yearsincurrentrole": 5,
    "yearssincelastpromotion": 1,
    "yearswithcurrmanager": 4,
}


MAIN_MODEL_URL = os.getenv("MAIN_MODEL_URL", "http://main-model:5000")
CANARY_MODEL_URL = os.getenv("CANARY_MODEL_URL", "http://canary-model:5001")
CANARY_TRAFFIC_PERCENT = float(os.getenv("CANARY_TRAFFIC_PERCENT", 20))


def load_registry() -> Dict[str, Any]:
    if not MODEL_REGISTRY_PATH.exists():
        return {}
    with MODEL_REGISTRY_PATH.open("r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return {}


def build_metrics_frame(registry: Dict[str, Any]) -> pd.DataFrame:
    records = []
    for model_key in ["main", "canary"]:
        entry = registry.get(model_key, {})
        for split in ["train", "test"]:
            metrics = entry.get(split)
            if not metrics:
                continue
            for metric_name, value in metrics.items():
                records.append(
                    {
                        "model": model_key,
                        "split": split,
                        "metric": metric_name,
                        "value": value,
                    }
                )
    return pd.DataFrame(records)


def render_metrics(metrics_df: pd.DataFrame) -> None:
    if metrics_df.empty:
        st.info("No metrics available. Train the models to populate the registry.")
        return
    fig = px.bar(
        metrics_df,
        x="metric",
        y="value",
        color="model",
        barmode="group",
        facet_col="split",
        title="Model performance (train/test)",
    )
    st.plotly_chart(fig, use_container_width=True)


def _call_model(url: str, payload: Dict[str, Any]) -> Tuple[float, int]:
    with httpx.Client(timeout=10) as client:
        response = client.post(f"{url}/predict", json=payload)
        response.raise_for_status()
        prediction = response.json().get("prediction", {})
        proba = float(prediction.get("churn_probability", 0.0))
        label = int(prediction.get("predicted_label", 0))
        return proba, label


def predict_remote(payload: Dict[str, Any]) -> pd.DataFrame:
    """Route a single request: 80% main, 20% canary by default."""
    feature_frame = prepare_features(payload)
    normalized_payload = feature_frame.iloc[0].to_dict()
    # Clean NaN/inf so JSON serialization succeeds
    for key, value in list(normalized_payload.items()):
        if value is None:
            continue
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            normalized_payload[key] = None
        elif pd.isna(value):
            normalized_payload[key] = None
        elif hasattr(value, "item"):
            normalized_payload[key] = value.item()

    routed_to = "canary" if random.random() * 100 < CANARY_TRAFFIC_PERCENT else "main"
    target_url = CANARY_MODEL_URL if routed_to == "canary" else MAIN_MODEL_URL

    rows = []
    try:
        proba, label = _call_model(target_url, normalized_payload)
        rows.append(
            {
                "model": routed_to,
                "churn_probability": proba,
                "predicted_label": label,
                "canary_split_percent": CANARY_TRAFFIC_PERCENT,
            }
        )
    except Exception as exc:  # pragma: no cover - UI feedback only
        rows.append(
            {
                "model": routed_to,
                "error": str(exc),
                "churn_probability": None,
                "predicted_label": None,
                "canary_split_percent": CANARY_TRAFFIC_PERCENT,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    st.set_page_config(page_title="Predicción de renuncia", page_icon="🧭", layout="wide")
    st.title("Predicción de renuncia de empleados")
    st.caption("Compara modelos main y canary, inspecciona métricas y realiza nuevas predicciones.")

    registry = load_registry()
    metrics_df = build_metrics_frame(registry)

    col_metrics, col_form = st.columns([1.2, 1])
    with col_metrics:
        st.subheader("Performance")
        render_metrics(metrics_df)

    with col_form:
        st.subheader("Predecir")
        st.write("Detalles del empleado:")
        with st.form("predict_form"):
            inputs: Dict[str, Any] = {}
            for field in CHURN_FEATURES:
                default = DEFAULT_SAMPLE.get(field)
                if isinstance(default, (int, float)):
                    inputs[field] = st.number_input(field, value=float(default), step=1.0)
                else:
                    inputs[field] = st.text_input(field, value=str(default or ""))
            submitted = st.form_submit_button("Ejecutar predicción")

        if submitted:
            results_df = predict_remote(inputs)
            routed_model = results_df.iloc[0]["model"]
            st.success(f"Predicción generada vía {routed_model} (canary {CANARY_TRAFFIC_PERCENT:.0f}% split)")
            st.dataframe(results_df, use_container_width=True)

            if results_df["churn_probability"].notna().any():
                chart = px.bar(
                    results_df.dropna(subset=["churn_probability"]),
                    x="model",
                    y="churn_probability",
                    color="model",
                    title="Churn probability by model",
                    range_y=[0, 1],
                    text_auto=True,
                )
                st.plotly_chart(chart, use_container_width=True)
            if "error" in results_df.columns and results_df["error"].notna().any():
                st.warning("Una o más llamadas al modelo fallaron; consulte la tabla para más detalles.")

    st.divider()
    st.markdown(
        "Los modelos se llaman a través de endpoints FastAPI (MAIN_MODEL_URL / CANARY_MODEL_URL). "
        "Las métricas se leen desde `artifacts/model_registry.json`."
    )


if __name__ == "__main__":
    main()
