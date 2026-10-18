import os
import sys
import joblib
import pandas as pd

from sqlalchemy import text
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

sys.path.insert(0, BASE_DIR)

from database import engine


MODEL_PATH = os.path.join(
    BASE_DIR,
    "risk_model.pkl"
)


FEATURE_COLUMNS = [
    "age",
    "conditions_count",
    "medications_count",
    "allergies_count",
    "encounters_count",
]


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print("Database se data load ho raha hai...")

    with engine.connect() as conn:

        patients = pd.read_sql(
            text("""
                SELECT
                    "Id",
                    "BIRTHDATE"
                FROM synthea_patients
            """),
            conn
        )

        conditions = pd.read_sql(
            text("""
                SELECT
                    "PATIENT",
                    "DESCRIPTION"
                FROM synthea_conditions
            """),
            conn
        )

        medications = pd.read_sql(
            text("""
                SELECT
                    "PATIENT",
                    "DESCRIPTION"
                FROM synthea_medications
            """),
            conn
        )

        allergies = pd.read_sql(
            text("""
                SELECT
                    "PATIENT",
                    "DESCRIPTION"
                FROM synthea_allergies
            """),
            conn
        )

        encounters = pd.read_sql(
            text("""
                SELECT
                    "PATIENT",
                    "ENCOUNTERCLASS"
                FROM synthea_encounters
            """),
            conn
        )

    print(f"Patients: {len(patients)}")
    print(f"Conditions: {len(conditions)}")
    print(f"Medications: {len(medications)}")
    print(f"Allergies: {len(allergies)}")
    print(f"Encounters: {len(encounters)}")

    return (
        patients,
        conditions,
        medications,
        encounters,
        allergies
    )


# ============================================================
# AGE CALCULATION
# ============================================================

def calculate_age(birthdate):

    if pd.isna(birthdate):
        return 0

    try:

        birthdate = pd.to_datetime(birthdate)

        today = pd.Timestamp.today()

        age = today.year - birthdate.year

        if (
            today.month,
            today.day
        ) < (
            birthdate.month,
            birthdate.day
        ):
            age -= 1

        return max(age, 0)

    except Exception:

        return 0


# ============================================================
# BUILD FEATURES
# ============================================================

def build_features(
    patients,
    conditions,
    medications,
    encounters,
    allergies
):

    print("Patient features build ho rahe hain...")

    features = patients[
        ["Id", "BIRTHDATE"]
    ].copy()

    # --------------------------------------------------------
    # Age
    # --------------------------------------------------------

    features["age"] = (
        features["BIRTHDATE"]
        .apply(calculate_age)
    )

    # --------------------------------------------------------
    # Conditions count
    # --------------------------------------------------------

    condition_counts = (
        conditions
        .groupby("PATIENT")
        .size()
        .rename("conditions_count")
    )

    # --------------------------------------------------------
    # Medication count
    # --------------------------------------------------------

    medication_counts = (
        medications
        .groupby("PATIENT")
        .size()
        .rename("medications_count")
    )

    # --------------------------------------------------------
    # Allergy count
    # --------------------------------------------------------

    allergy_counts = (
        allergies
        .groupby("PATIENT")
        .size()
        .rename("allergies_count")
    )

    # --------------------------------------------------------
    # Encounter count
    # --------------------------------------------------------

    encounter_counts = (
        encounters
        .groupby("PATIENT")
        .size()
        .rename("encounters_count")
    )

    # --------------------------------------------------------
    # Merge conditions
    # --------------------------------------------------------

    features = features.merge(
        condition_counts,
        left_on="Id",
        right_index=True,
        how="left"
    )

    # --------------------------------------------------------
    # Merge medications
    # --------------------------------------------------------

    features = features.merge(
        medication_counts,
        left_on="Id",
        right_index=True,
        how="left"
    )

    # --------------------------------------------------------
    # Merge allergies
    # --------------------------------------------------------

    features = features.merge(
        allergy_counts,
        left_on="Id",
        right_index=True,
        how="left"
    )

    # --------------------------------------------------------
    # Merge encounters
    # --------------------------------------------------------

    features = features.merge(
        encounter_counts,
        left_on="Id",
        right_index=True,
        how="left"
    )

    # --------------------------------------------------------
    # Fill missing counts
    # --------------------------------------------------------

    count_columns = [
        "conditions_count",
        "medications_count",
        "allergies_count",
        "encounters_count"
    ]

    for column in count_columns:

        features[column] = (
            features[column]
            .fillna(0)
            .astype(int)
        )

    # ========================================================
    # RESEARCH / DEMO TARGET
    # ========================================================

    features["target"] = (
        (
            (features["conditions_count"] >= 5)
            |
            (features["medications_count"] >= 5)
            |
            (features["encounters_count"] >= 10)
        )
        .astype(int)
    )

    features = features.set_index("Id")

    print("Feature columns:")
    print(features.columns.tolist())

    return features


# ============================================================
# TRAIN MODEL
# ============================================================

def train_model():

    (
        patients,
        conditions,
        medications,
        encounters,
        allergies
    ) = load_data()

    features = build_features(
        patients,
        conditions,
        medications,
        encounters,
        allergies
    )

    X = features[FEATURE_COLUMNS]

    y = features["target"]

    # ========================================================
    # HANDLE TARGET
    # ========================================================

    if y.nunique() < 2:

        print(
            "Warning: Target mein sirf ek class mili."
        )

        score = (
            features["conditions_count"]
            +
            features["medications_count"]
            +
            features["encounters_count"]
        )

        threshold = score.median()

        y = (
            score > threshold
        ).astype(int)

    # --------------------------------------------------------
    # If still only one class
    # --------------------------------------------------------

    if y.nunique() < 2:

        y = pd.Series(
            [
                0 if i < len(y) // 2 else 1
                for i in range(len(y))
            ],
            index=y.index
        )

    print(
        f"Training rows: {len(X)}"
    )

    print(
        f"Target classes: {y.value_counts().to_dict()}"
    )

    # ========================================================
    # MODEL
    # ========================================================

    pipeline = Pipeline(
        steps=[
            (
                "scaler",
                StandardScaler()
            ),
            (
                "model",
                LogisticRegression(
                    max_iter=1000
                )
            )
        ]
    )

    pipeline.fit(
        X,
        y
    )

    # ========================================================
    # SAVE MODEL
    # ========================================================

    bundle = {
        "model": pipeline,
        "scaler": None,
        "needs_scaler": False,
        "feature_columns": FEATURE_COLUMNS,
        "feature_cols": FEATURE_COLUMNS
    }

    joblib.dump(
        bundle,
        MODEL_PATH
    )

    print()
    print(
        "SUCCESS: risk_model.pkl generate ho gayi."
    )

    print(
        f"Saved at: {MODEL_PATH}"
    )

    return bundle


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    if not os.path.exists(MODEL_PATH):

        print(
            "Warning: risk_model.pkl not found."
        )

        return None

    return joblib.load(
        MODEL_PATH
    )


# ============================================================
# LIST ALL PATIENT IDS
# ============================================================

def list_patient_ids():

    try:

        with engine.connect() as conn:

            patients = pd.read_sql(
                text("""
                    SELECT
                        "Id"
                    FROM synthea_patients
                    ORDER BY "Id"
                """),
                conn
            )

        return (
            patients["Id"]
            .astype(str)
            .tolist()
        )

    except Exception as e:

        print(
            "Patient ID error:",
            e
        )

        return []


# ============================================================
# GET FEATURES FOR PATIENT
# ============================================================

def get_features_for_patient(patient_id):

    try:

        (
            patients,
            conditions,
            medications,
            encounters,
            allergies
        ) = load_data()

        features = build_features(
            patients,
            conditions,
            medications,
            encounters,
            allergies
        )

        patient_id = str(patient_id)

        matching = features[
            features.index.astype(str)
            == patient_id
        ]

        if matching.empty:

            return None

        row = matching.iloc[0]

        return {
            "age": int(
                row["age"]
            ),

            "conditions_count": int(
                row["conditions_count"]
            ),

            "medications_count": int(
                row["medications_count"]
            ),

            "allergies_count": int(
                row["allergies_count"]
            ),

            "encounters_count": int(
                row["encounters_count"]
            )
        }

    except Exception as e:

        print(
            "Feature error:",
            e
        )

        return None


# ============================================================
# GET PATIENT RECORD
# ============================================================

def get_patient_record(patient_id):

    try:

        (
            patients,
            conditions,
            medications,
            encounters,
            allergies
        ) = load_data()

        patient_id = str(patient_id)

        patient = patients[
            patients["Id"].astype(str)
            == patient_id
        ]

        if patient.empty:

            return None

        row = patient.iloc[0]

        patient_conditions = conditions[
            conditions["PATIENT"].astype(str)
            == patient_id
        ]

        patient_medications = medications[
            medications["PATIENT"].astype(str)
            == patient_id
        ]

        patient_allergies = allergies[
            allergies["PATIENT"].astype(str)
            == patient_id
        ]

        patient_encounters = encounters[
            encounters["PATIENT"].astype(str)
            == patient_id
        ]

        return {

            "Id": patient_id,

            "BIRTHDATE": row.get(
                "BIRTHDATE"
            ),

            "age": calculate_age(
                row.get("BIRTHDATE")
            ),

            "conditions_count": len(
                patient_conditions
            ),

            "medications_count": len(
                patient_medications
            ),

            "allergies_count": len(
                patient_allergies
            ),

            "encounters_count": len(
                patient_encounters
            )
        }

    except Exception as e:

        print(
            "Patient record error:",
            e
        )

        return None


# ============================================================
# PREDICT RISK
# ============================================================

def predict_risk_for_patient(patient_id):

    try:

        bundle = load_model()

        if bundle is None:

            return None

        features = get_features_for_patient(
            patient_id
        )

        if features is None:

            return None

        X = pd.DataFrame(
            [features],
            columns=FEATURE_COLUMNS
        )

        model = bundle["model"]

        risk_score = float(
            model.predict_proba(X)[0][1]
        )

        # ====================================================
        # RISK LEVEL
        # ====================================================

        if risk_score < 0.40:

            risk_level = "Low"

        elif risk_score < 0.70:

            risk_level = "Medium"

        else:

            risk_level = "High"

        # ====================================================
        # EXPLANATION
        # ====================================================

        explanation = []

        if features["age"] >= 65:

            explanation.append(
                "Older patient age"
            )

        if features["conditions_count"] >= 5:

            explanation.append(
                "Multiple medical conditions"
            )

        if features["medications_count"] >= 5:

            explanation.append(
                "Multiple medications"
            )

        if features["allergies_count"] >= 2:

            explanation.append(
                "Multiple allergies"
            )

        if features["encounters_count"] >= 10:

            explanation.append(
                "Frequent healthcare encounters"
            )

        if not explanation:

            explanation.append(
                "No major risk factors detected"
            )

        return {

            "patient_id": str(
                patient_id
            ),

            "risk_score": round(
                risk_score,
                4
            ),

            "risk_percentage": round(
                risk_score * 100,
                2
            ),

            "risk_level": risk_level,

            "explanation": explanation,

            "features": features
        }

    except Exception as e:

        print(
            "Prediction error:",
            e
        )

        return None


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print(
        "Clinical AI Risk Engine"
    )

    print(
        "=" * 40
    )

    train_model()
# ============================================================
# GET ONE PATIENT RISK
# ============================================================

def get_patient_risk(patient_id):

    patient_id = str(patient_id).strip()

    if not patient_id:
        return None

    patient = get_patient_record(patient_id)

    if patient is None:
        return None

    model = load_model()

    features = pd.DataFrame([
        {
            column: patient[column]
            for column in FEATURE_COLUMNS
        }
    ])

    probability = model.predict_proba(features)[0][1]

    risk_level = get_risk_level(probability)

    return {
        "patient_id": patient_id,
        "age": int(patient["age"]),
        "conditions_count": int(patient["conditions_count"]),
        "medications_count": int(patient["medications_count"]),
        "allergies_count": int(patient["allergies_count"]),
        "encounters_count": int(patient["encounters_count"]),
        "risk_score": round(probability * 100, 2),
        "risk_level": risk_level
    }

