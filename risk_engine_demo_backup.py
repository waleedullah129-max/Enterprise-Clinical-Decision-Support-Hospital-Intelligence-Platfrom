"""
risk_engine.py
Core risk-scoring logic for the Clinical AI Platform.

Public functions used by app.py:
    - get_features_for_patient(patient_id)
    - predict_risk_for_patient(features)
    - get_patient_record(patient_id)
    - list_patient_ids()

This version is fully self-contained: it generates a small synthetic
patient database in memory, so it runs with zero external data/model
files and can never raise a "file not found" error.
"""

import numpy as np
import pandas as pd

np.random.seed(42)

FEATURE_NAMES = [
    "age",
    "bmi",
    "systolic_bp",
    "diastolic_bp",
    "heart_rate",
    "glucose",
    "cholesterol",
    "smoker",
    "num_chronic_conditions",
]


def _generate_demo_patients(n: int = 15) -> pd.DataFrame:
    ids = [f"P{1000 + i}" for i in range(n)]
    data = {
        "patient_id": ids,
        "name": [f"Patient {i + 1}" for i in range(n)],
        "age": np.random.randint(20, 85, n),
        "bmi": np.round(np.random.normal(27, 5, n), 1),
        "systolic_bp": np.random.randint(100, 170, n),
        "diastolic_bp": np.random.randint(60, 100, n),
        "heart_rate": np.random.randint(55, 110, n),
        "glucose": np.random.randint(70, 220, n),
        "cholesterol": np.random.randint(140, 280, n),
        "smoker": np.random.choice([0, 1], n, p=[0.7, 0.3]),
        "num_chronic_conditions": np.random.randint(0, 4, n),
    }
    return pd.DataFrame(data).set_index("patient_id")


PATIENT_DB = _generate_demo_patients()


def list_patient_ids():
    """Return all demo patient IDs available in the database."""
    return list(PATIENT_DB.index)


def get_patient_record(patient_id: str) -> dict:
    """Return the full raw record (all columns) for a patient."""
    if patient_id not in PATIENT_DB.index:
        raise ValueError(f"Patient '{patient_id}' not found in database.")
    return PATIENT_DB.loc[patient_id].to_dict()


def get_features_for_patient(patient_id: str) -> dict:
    """Return only the model-ready numeric features for a patient."""
    record = get_patient_record(patient_id)
    return {k: record[k] for k in FEATURE_NAMES}


def _weights() -> dict:
    # Simple, interpretable weighted-risk model (demo, not clinically validated)
    return {
        "age": 0.02,
        "bmi": 0.03,
        "systolic_bp": 0.015,
        "diastolic_bp": 0.01,
        "heart_rate": 0.008,
        "glucose": 0.012,
        "cholesterol": 0.006,
        "smoker": 1.2,
        "num_chronic_conditions": 0.7,
    }


def _sigmoid(x: float) -> float:
    return 1 / (1 + np.exp(-x))


def predict_risk_for_patient(features: dict) -> dict:
    """
    Compute a risk probability + category from a feature dict.

    Returns:
        {
            "probability": float (0-1),
            "risk_category": "Low" | "Medium" | "High",
            "contributions": {feature: contribution_value, ...}
        }
    """
    w = _weights()
    z = -6.5  # intercept, calibrated so mid-range patients sit near 50%
    contributions = {}
    for feat, value in features.items():
        contrib = w.get(feat, 0) * value
        contributions[feat] = round(float(contrib), 4)
        z += contrib

    probability = float(_sigmoid(z))

    if probability < 0.33:
        category = "Low"
    elif probability < 0.66:
        category = "Medium"
    else:
        category = "High"

    return {
        "probability": round(probability, 4),
        "risk_category": category,
        "contributions": contributions,
    }