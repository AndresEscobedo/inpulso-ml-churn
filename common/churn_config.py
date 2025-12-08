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
    # kept from the notebook selection after dropping EmployeeNumber, Over18, StandardHours
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

# Notebook-driven one-hot list (drop_first=True in notebook via pd.get_dummies)
CATEGORICAL_DUMMIES: List[str] = [
    "businesstravel",
    "department",
    "education",
    "educationfield",
    "environmentsatisfaction",
    "gender",
    "jobinvolvement",
    "joblevel",
    "jobrole",
    "maritalstatus",
]

# Binary mappings performed before the encoder
BINARY_MAPPINGS: List[str] = ["overtime", "disobediencerules"]

NUMERIC_FEATURES: List[str] = [
    feature for feature in CHURN_FEATURES if feature not in CATEGORICAL_DUMMIES
]
