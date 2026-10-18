"""
run_medication_safety.py
==========================
medication_safety.py ke rule-based checks ko REAL Synthea data (PostgreSQL)
par chalata hai — hardcoded P001/P002 test data ki jagah.

Chalane ka tarika:
    python run_medication_safety.py
"""

import pandas as pd

from database import engine
from medication_safety import run_medication_safety_check


def load_real_data():
    print("Synthea medications aur allergies PostgreSQL se load ho rahe hain...")

    medications = pd.read_sql(
        'SELECT "PATIENT" AS patient_id, "DESCRIPTION" AS medication, "START", "STOP" '
        "FROM synthea_medications",
        engine,
    )
    allergies = pd.read_sql(
        'SELECT "PATIENT" AS patient_id, "DESCRIPTION" AS allergy '
        "FROM synthea_allergies",
        engine,
    )

    print(f"medications: {len(medications)} rows | allergies: {len(allergies)} rows")
    return medications, allergies


if __name__ == "__main__":
    medications, allergies = load_real_data()

    results = run_medication_safety_check(medications, allergies)

    print("\n=== DRUG-DRUG INTERACTIONS ===")
    print(f"Total findings: {len(results['drug_interactions'])}")
    if not results["drug_interactions"].empty:
        print(results["drug_interactions"].to_string(index=False))

    print("\n=== ALLERGY CONFLICTS ===")
    print(f"Total findings: {len(results['allergy_conflicts'])}")
    if not results["allergy_conflicts"].empty:
        print(results["allergy_conflicts"].to_string(index=False))

    print("\n=== DUPLICATE PRESCRIPTIONS ===")
    print(f"Total findings: {len(results['duplicate_prescriptions'])}")
    if not results["duplicate_prescriptions"].empty:
        print(results["duplicate_prescriptions"].head(20).to_string(index=False))

    # Results ko CSV mein bhi save kar dete hain, dashboard/report ke liye
    results["drug_interactions"].to_csv("drug_interactions_report.csv", index=False)
    results["allergy_conflicts"].to_csv("allergy_conflicts_report.csv", index=False)
    results["duplicate_prescriptions"].to_csv(
        "duplicate_prescriptions_report.csv", index=False
    )
    print("\n✅ Reports save ho gayi: drug_interactions_report.csv, "
          "allergy_conflicts_report.csv, duplicate_prescriptions_report.csv")
