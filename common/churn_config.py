"""Centralized churn model settings shared across training and serving code."""
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
MAIN_MODEL_ARTIFACT = ARTIFACT_DIR / "main_model.joblib"
CANARY_MODEL_ARTIFACT = ARTIFACT_DIR / "canary_model.joblib"
MODEL_REGISTRY_PATH = ARTIFACT_DIR / "model_registry.json"
DATA_PATH = PROJECT_ROOT / "HR_Employee_Attrition_Dataset con faltantes.csv"
TARGET_COLUMN = "attrition"
RANDOM_STATE = 42

CHURN_FEATURES: List[str] = [
    "age",
    "businesstravel",
    "dailyrate",
    "department",
    "distancefromhome",
    "education",
    "educationfield",
    "environmentsatisfaction",
    "gender",
    "hourlyrate",
    "jobinvolvement",
    "joblevel",
    "jobrole",
    "disobediencerules",
    "jobsatisfaction",
    "maritalstatus",
    "monthlyincome",
    "monthlyrate",
    "numcompaniesworked",
    "overtime",
    "percentsalaryhike",
    "performancerating",
    "relationshipsatisfaction",
    "stockoptionlevel",
    "totalworkingyears",
    "trainingtimeslastyear",
    "worklifebalance",
    "yearsatcompany",
    "yearsincurrentrole",
    "yearssincelastpromotion",
    "yearswithcurrmanager",
]

CATEGORICAL_FEATURES: List[str] = [
    "businesstravel",
    "department",
    "educationfield",
    "gender",
    "jobrole",
    "disobediencerules",
    "maritalstatus",
    "overtime",
]

NUMERIC_FEATURES: List[str] = [
    feature for feature in CHURN_FEATURES if feature not in CATEGORICAL_FEATURES
]
