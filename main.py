from fastapi import FastAPI, HTTPException
from sqlalchemy import text
from src.database import engine
from src.risk_engine import predict_risk_for_patient

import pandas as pd
from datetime import datetime


app = FastAPI(title="Clinical AI Platform API")


# ============================================================
# HOME
# ============================================================

@app.get("/")
def home():
    return {
        "message": "Clinical AI Platform API chal raha hai!"
    }


# ============================================================
# MIMIC PATIENT
# ============================================================

@app.get("/mimic/patients/{subject_id}")
def get_mimic_patient(subject_id: int):

    query = text("""
        SELECT *
        FROM mimic_patients
        WHERE subject_id = :sid
    """)

    with engine.connect() as conn:
        row = conn.execute(
            query,
            {"sid": subject_id}
        ).mappings().first()

    if row is None:
        raise HTTPException(
            status_code=404,
            detail="Patient nahi mila"
        )

    return dict(row)


# ============================================================
# MIMIC DIAGNOSES
# ============================================================

@app.get("/mimic/patients/{subject_id}/diagnoses")
def get_mimic_diagnoses(subject_id: int):

    query = text("""
        SELECT *
        FROM mimic_diagnoses
        WHERE subject_id = :sid
    """)

    with engine.connect() as conn:
        rows = conn.execute(
            query,
            {"sid": subject_id}
        ).mappings().all()

    return [dict(row) for row in rows]


# ============================================================
# MIMIC ICU STAYS
# ============================================================

@app.get("/mimic/patients/{subject_id}/icustays")
def get_mimic_icustays(subject_id: int):

    query = text("""
        SELECT *
        FROM mimic_icustays
        WHERE subject_id = :sid
    """)

    with engine.connect() as conn:
        rows = conn.execute(
            query,
            {"sid": subject_id}
        ).mappings().all()

    return [dict(row) for row in rows]


# ============================================================
# SYNTHEA PATIENTS
# ============================================================

@app.get("/synthea/patients")
def get_synthea_patients(limit: int = 10):

    query = text("""
        SELECT *
        FROM synthea_patients
        LIMIT :lim
    """)

    with engine.connect() as conn:
        rows = conn.execute(
            query,
            {"lim": limit}
        ).mappings().all()

    return [dict(row) for row in rows]


# ============================================================
# SYNTHEA PATIENT RISK SCORE
# ============================================================

@app.get("/synthea/patients/{patient_id}/risk-score")
def get_patient_risk_score(patient_id: str):

    patient_query = text("""
        SELECT "BIRTHDATE", "DEATHDATE"
        FROM synthea_patients
        WHERE "Id" = :pid
    """)

    with engine.connect() as conn:
        row = conn.execute(
            patient_query,
            {"pid": patient_id}
        ).fetchone()

    if row is None:
        raise HTTPException(
            status_code=404,
            detail="Patient nahi mila"
        )

    birthdate = pd.to_datetime(row[0], errors="coerce")

    if row[1]:
        reference_date = pd.to_datetime(
            row[1],
            errors="coerce"
        )
    else:
        reference_date = pd.Timestamp(datetime.now())

    age = round(
        (reference_date - birthdate).days / 365.25,
        1
    )

    counts_query = text("""
        SELECT
            (
                SELECT COUNT(*)
                FROM synthea_conditions
                WHERE "PATIENT" = :pid
            ) AS conditions_count,

            (
                SELECT COUNT(*)
                FROM synthea_medications
                WHERE "PATIENT" = :pid
            ) AS medications_count,

            (
                SELECT COUNT(*)
                FROM synthea_allergies
                WHERE "PATIENT" = :pid
            ) AS allergies_count,

            (
                SELECT COUNT(*)
                FROM synthea_encounters
                WHERE "PATIENT" = :pid
            ) AS encounters_count
    """)

    with engine.connect() as conn:
        counts = conn.execute(
            counts_query,
            {"pid": patient_id}
        ).fetchone()

    patient_features = {
        "age": age,
        "conditions_count": counts[0],
        "medications_count": counts[1],
        "allergies_count": counts[2],
        "encounters_count": counts[3],
    }

    try:
        result = predict_risk_for_patient(
            patient_features
        )

    except FileNotFoundError:
        raise HTTPException(
            status_code=500,
            detail=(
                "Risk model nahi mili. "
                "Pehle risk_engine.py run karein."
            )
        )

    return {
        "patient_id": patient_id,
        **patient_features,
        **result
    }


# ============================================================
# SYNTHEA CONDITIONS
# ============================================================

@app.get("/synthea/patients/{patient_id}/conditions")
def get_synthea_conditions(patient_id: str):

    query = text("""
        SELECT *
        FROM synthea_conditions
        WHERE "PATIENT" = :pid
    """)

    with engine.connect() as conn:
        rows = conn.execute(
            query,
            {"pid": patient_id}
        ).mappings().all()

    return [dict(row) for row in rows]


# ============================================================
# DASHBOARD STATISTICS
# ============================================================

@app.get("/dashboard/stats")
def get_dashboard_stats():

    query = text("""
        SELECT
            (SELECT COUNT(*) FROM synthea_patients)
                AS total_synthea_patients,

            (SELECT COUNT(*) FROM mimic_patients)
                AS total_mimic_patients,

            (SELECT COUNT(*) FROM mimic_admissions)
                AS total_admissions,

            (SELECT COUNT(*) FROM mimic_icustays)
                AS total_icu_stays,

            (SELECT COUNT(*) FROM mimic_labevents)
                AS total_lab_events
    """)

    with engine.connect() as conn:
        row = conn.execute(query).mappings().first()

    return dict(row)


# ============================================================
# RISK SCORES
# ============================================================

@app.get("/risk-scores")
def get_risk_scores():

    query = text("""
        SELECT
            "Id" AS patient_id,
            "BIRTHDATE",
            "DEATHDATE"
        FROM synthea_patients
        LIMIT 100
    """)

    with engine.connect() as conn:
        patients = conn.execute(query).mappings().all()

    results = []

    for patient in patients:

        birthdate = pd.to_datetime(
            patient["BIRTHDATE"],
            errors="coerce"
        )

        if patient["DEATHDATE"]:
            reference_date = pd.to_datetime(
                patient["DEATHDATE"],
                errors="coerce"
            )
        else:
            reference_date = pd.Timestamp(datetime.now())

        age = round(
            (reference_date - birthdate).days / 365.25,
            1
        )

        counts_query = text("""
            SELECT
                (SELECT COUNT(*)
                 FROM synthea_conditions
                 WHERE "PATIENT" = :pid),

                (SELECT COUNT(*)
                 FROM synthea_medications
                 WHERE "PATIENT" = :pid),

                (SELECT COUNT(*)
                 FROM synthea_allergies
                 WHERE "PATIENT" = :pid),

                (SELECT COUNT(*)
                 FROM synthea_encounters
                 WHERE "PATIENT" = :pid)
        """)

        with engine.connect() as conn:
            counts = conn.execute(
                counts_query,
                {"pid": patient["patient_id"]}
            ).fetchone()

        features = {
            "age": age,
            "conditions_count": counts[0],
            "medications_count": counts[1],
            "allergies_count": counts[2],
            "encounters_count": counts[3],
        }

        try:
            risk_result = predict_risk_for_patient(features)

            results.append({
                "patient_id": patient["patient_id"],
                **features,
                **risk_result
            })

        except FileNotFoundError:
            results.append({
                "patient_id": patient["patient_id"],
                "risk_score": None,
                "risk_percentage": None,
                "risk_level": "Model Not Trained",
                "risk_reasons": []
            })

    return {
        "total": len(results),
        "risk_scores": results
    }