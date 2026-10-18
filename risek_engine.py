import os
import sys
import joblib
import pandas as pd

from sqlalchemy import text
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)

if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from database import engine


MODEL_PATH = os.path.join(BASE_DIR, "risk_model.pkl")


# ============================================================
# CONSTANTS
# ============================================================

FEATURE_COLUMNS = [
    "age",
    "conditions_count",
    "medications_count",
    "allergies_count",
    "encounters_count",
]

MEDIUM_THRESHOLD = 0.40
HIGH_THRESHOLD = 0.70


# ============================================================
# DATABASE CONNECTION
# ============================================================

def test_database_connection():

    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))

        return True

    except Exception as e:
        print(f"Database connection error: {e}")
        return False


# ============================================================
# GET ALL PATIENT FEATURES
# ============================================================

def get_patient_features():

    query = text("""
        SELECT
            p."Id" AS patient_id,

            EXTRACT(
                YEAR FROM AGE(
                    CURRENT_DATE,
                    TO_DATE(p."BIRTHDATE", 'YYYY-MM-DD')
                )
            ) AS age,

            COUNT(DISTINCT c."Id") AS conditions_count,
            COUNT(DISTINCT m."Id") AS medications_count,
            COUNT(DISTINCT a."Id") AS allergies_count,
            COUNT(DISTINCT e."Id") AS encounters_count

        FROM synthea_patients p

        LEFT JOIN synthea_conditions c
            ON p."Id" = c."PATIENT"

        LEFT JOIN synthea_medications m
            ON p."Id" = m."PATIENT"

        LEFT JOIN synthea_allergies a
            ON p."Id" = a."PATIENT"

        LEFT JOIN synthea_encounters e
            ON p."Id" = e."PATIENT"

        GROUP BY
            p."Id",
            p."BIRTHDATE"

        ORDER BY
            p."Id"
    """)

    with engine.connect() as conn:
        df = pd.read_sql(query, conn)

    if df.empty:
        raise ValueError(
            "No patients found in synthea_patients table."
        )

    for column in FEATURE_COLUMNS:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        ).fillna(0)

    return df


# ============================================================
# CREATE DEMO RISK LABEL
# ============================================================

def create_labels(df):

    df = df.copy()

    score = (
        (df["age"] >= 65).astype(int)
        + (df["conditions_count"] >= 5).astype(int)
        + (df["medications_count"] >= 5).astype(int)
        + (df["encounters_count"] >= 10).astype(int)
    )

    df["high_risk"] = (
        score >= 2
    ).astype(int)

    return df


# ============================================================
# CREATE MODEL
# ============================================================

def create_model():

    return Pipeline([
        (
            "scaler",
            StandardScaler()
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                random_state=42,
                class_weight="balanced"
            )
        )
    ])


# ============================================================
# GET RISK LEVEL
# ============================================================

def get_risk_level(probability):

    probability = float(probability)

    if probability >= HIGH_THRESHOLD:
        return "High"

    elif probability >= MEDIUM_THRESHOLD:
        return "Medium"

    return "Low"


# ============================================================
# TRAIN MODEL
# ============================================================

def train_model():

    print("\n======================================")
    print("Clinical AI - Risk Engine")
    print("======================================")

    print("\n[1/5] Checking database...")

    if not test_database_connection():
        raise ConnectionError(
            "Cannot connect to PostgreSQL database."
        )

    print("Database connection successful.")

    print("\n[2/5] Loading patient data...")

    df = get_patient_features()

    print(
        f"Patients loaded: {len(df)}"
    )

    print("\nPreparing features...")

    for column in FEATURE_COLUMNS:

        if column not in df.columns:
            raise ValueError(
                f"Missing feature: {column}"
            )

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        ).fillna(0)

    df = create_labels(df)

    print("\nRisk distribution:")
    print(
        df["high_risk"]
        .value_counts()
        .sort_index()
    )

    X = df[FEATURE_COLUMNS]
    y = df["high_risk"]

    if y.nunique() < 2:
        raise ValueError(
            "Training data contains only one risk class."
        )

    print(
        "\n[3/5] Training Logistic Regression..."
    )

    model = create_model()
    model.fit(X, y)

    print(
        "Model training completed."
    )

    print("\n[4/5] Saving model...")

    # IMPORTANT:
    # Save the sklearn Pipeline directly.
    # Do NOT save a dictionary bundle.
    joblib.dump(
        model,
        MODEL_PATH
    )

    print(
        f"Model saved to:\n{MODEL_PATH}"
    )

    print(
        "\n[5/5] Creating patient risk scores..."
    )

    probabilities = model.predict_proba(X)[:, 1]

    df["risk_score"] = probabilities

    df["risk_level"] = df[
        "risk_score"
    ].apply(
        get_risk_level
    )

    results = df[
        [
            "patient_id",
            "risk_score",
            "risk_level"
        ]
    ].copy()

    results["risk_score"] = (
        results["risk_score"] * 100
    ).round(2)

    results.to_sql(
        "patient_risk_scores",
        engine,
        if_exists="replace",
        index=False
    )

    print(
        "\nRisk scores saved to PostgreSQL."
    )

    print("\n======================================")
    print("RISK ENGINE COMPLETED")
    print("======================================")

    print("\nSample risk scores:")

    print(
        results
        .head(10)
        .to_string(index=False)
    )

    return results


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    if not os.path.exists(MODEL_PATH):

        print(
            "Warning: risk_model.pkl not found."
        )

        return None

    loaded = joblib.load(MODEL_PATH)

    # New format: direct sklearn model
    if hasattr(loaded, "predict_proba"):
        return loaded

    # Old format: dictionary bundle
    # Kept for compatibility with an older model file.
    if isinstance(loaded, dict):

        if "model" in loaded:

            model = loaded["model"]

            if hasattr(model, "predict_proba"):
                return model

    raise TypeError(
        f"Unsupported risk model format: {type(loaded)}"
    )


# ============================================================
# GET ONE PATIENT RECORD
# ============================================================

def get_patient_record(patient_id):

    query = text("""
        SELECT
            p."Id" AS patient_id,

            EXTRACT(
                YEAR FROM AGE(
                    CURRENT_DATE,
                    TO_DATE(p."BIRTHDATE", 'YYYY-MM-DD')
                )
            ) AS age,

            COUNT(DISTINCT c."Id") AS conditions_count,
            COUNT(DISTINCT m."Id") AS medications_count,
            COUNT(DISTINCT a."Id") AS allergies_count,
            COUNT(DISTINCT e."Id") AS encounters_count

        FROM synthea_patients p

        LEFT JOIN synthea_conditions c
            ON p."Id" = c."PATIENT"

        LEFT JOIN synthea_medications m
            ON p."Id" = m."PATIENT"

        LEFT JOIN synthea_allergies a
            ON p."Id" = a."PATIENT"

        LEFT JOIN synthea_encounters e
            ON p."Id" = e."PATIENT"

        WHERE p."Id" = :patient_id

        GROUP BY
            p."Id",
            p."BIRTHDATE"
    """)

    with engine.connect() as conn:

        df = pd.read_sql(
            query,
            conn,
            params={
                "patient_id": patient_id
            }
        )

    if df.empty:
        return None

    for column in FEATURE_COLUMNS:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        ).fillna(0)

    return df.iloc[0].to_dict()


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

    # If model doesn't exist, train it automatically.
    if model is None:

        print(
            "Risk model missing. Training a new model..."
        )

        train_model()

        model = load_model()

    if model is None:

        raise RuntimeError(
            "Risk model could not be loaded."
        )

    features = pd.DataFrame([
        {
            column: patient[column]
            for column in FEATURE_COLUMNS
        }
    ])

    probability = float(
        model.predict_proba(features)[0][1]
    )

    risk_level = get_risk_level(
        probability
    )

    return {
        "patient_id": patient_id,
        "age": int(patient["age"]),
        "conditions_count": int(
            patient["conditions_count"]
        ),
        "medications_count": int(
            patient["medications_count"]
        ),
        "allergies_count": int(
            patient["allergies_count"]
        ),
        "encounters_count": int(
            patient["encounters_count"]
        ),
        "risk_score": round(
            probability * 100,
            2
        ),
        "risk_level": risk_level
    }


# ============================================================
# LIST PATIENT IDS
# ============================================================

def list_patient_ids():

    query = text("""
        SELECT
            "Id" AS patient_id
        FROM synthea_patients
        ORDER BY "Id"
    """)

    with engine.connect() as conn:

        df = pd.read_sql(
            query,
            conn
        )

    if df.empty:
        return []

    return (
        df["patient_id"]
        .astype(str)
        .tolist()
    )


# ============================================================
# GET FEATURES FOR ONE PATIENT
# ============================================================

def get_features_for_patient(patient_id):

    patient = get_patient_record(
        patient_id
    )

    if patient is None:
        return None

    return {

        "age": int(
            patient["age"]
        ),

        "conditions_count": int(
            patient["conditions_count"]
        ),

        "medications_count": int(
            patient["medications_count"]
        ),

        "allergies_count": int(
            patient["allergies_count"]
        ),

        "encounters_count": int(
            patient["encounters_count"]
        )
    }


# ============================================================
# GET RISK SUMMARY
# ============================================================

def get_risk_summary():

    query = text("""
        SELECT
            risk_level,
            COUNT(*) AS patient_count
        FROM patient_risk_scores
        GROUP BY risk_level
        ORDER BY risk_level
    """)

    try:

        with engine.connect() as conn:

            df = pd.read_sql(
                query,
                conn
            )

        return df

    except Exception:

        return pd.DataFrame(
            columns=[
                "risk_level",
                "patient_count"
            ]
        )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    try:

        results = train_model()

        print(
            "\nRisk engine is ready."
        )

        print(
            "\nTotal risk scores:"
        )

        print(
            len(results)
        )

        print(
            "\nRisk level summary:"
        )

        print(
            results["risk_level"]
            .value_counts()
            .to_string()
        )

    except Exception as e:

        print("\n======================================")
        print("RISK ENGINE ERROR")
        print("======================================")

        print(e)

        raise