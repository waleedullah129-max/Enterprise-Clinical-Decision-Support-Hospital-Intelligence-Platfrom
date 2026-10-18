"""
Medication Safety Engine

Detects:
1. Drug-drug interactions
2. Allergy conflicts
3. Duplicate prescriptions

This is a simple rule-based clinical decision-support module.
It is not a replacement for a licensed clinician or pharmacist.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Optional

import pandas as pd


# ============================================================
# 1. BASIC DRUG-DRUG INTERACTION RULES
# ============================================================

DRUG_INTERACTIONS = {
    frozenset({"warfarin", "aspirin"}): {
        "severity": "High",
        "message": (
            "Warfarin and aspirin together may increase bleeding risk."
        ),
    },
    frozenset({"warfarin", "ibuprofen"}): {
        "severity": "High",
        "message": (
            "Warfarin and ibuprofen may increase bleeding risk."
        ),
    },
    frozenset({"warfarin", "naproxen"}): {
        "severity": "High",
        "message": (
            "Warfarin and naproxen may increase bleeding risk."
        ),
    },
    frozenset({"metformin", "contrast dye"}): {
        "severity": "High",
        "message": (
            "Metformin with contrast dye requires clinical review."
        ),
    },
    frozenset({"simvastatin", "clarithromycin"}): {
        "severity": "High",
        "message": (
            "Clarithromycin can increase simvastatin-related toxicity risk."
        ),
    },
    frozenset({"lisinopril", "potassium"}): {
        "severity": "Moderate",
        "message": (
            "Lisinopril and potassium may increase hyperkalemia risk."
        ),
    },
    frozenset({"ciprofloxacin", "warfarin"}): {
        "severity": "Moderate",
        "message": (
            "Ciprofloxacin may affect warfarin anticoagulation."
        ),
    },
}


# ============================================================
# 2. ALLERGY CONFLICT RULES
# ============================================================

ALLERGY_CONFLICTS = {
    "penicillin": {
        "amoxicillin",
        "ampicillin",
        "penicillin",
        "amoxicillin-clavulanate",
    },
    "sulfa": {
        "sulfamethoxazole",
        "trimethoprim-sulfamethoxazole",
        "sulfasalazine",
    },
    "nsaid": {
        "ibuprofen",
        "naproxen",
        "diclofenac",
        "aspirin",
    },
    "aspirin": {
        "aspirin",
    },
    "ibuprofen": {
        "ibuprofen",
    },
}


# ============================================================
# 3. HELPER FUNCTIONS
# ============================================================

def normalize(value: Any) -> str:
    """
    Convert a value into a comparable lowercase string.
    """

    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass

    return str(value).strip().lower()


def find_column(
    df: pd.DataFrame,
    possible_names: List[str],
) -> Optional[str]:
    """
    Find the first matching column from possible names.
    Matching is case-insensitive.
    """

    if df is None or df.empty:
        return None

    normalized_columns = {
        str(column).strip().lower(): column
        for column in df.columns
    }

    for name in possible_names:
        if name.lower() in normalized_columns:
            return normalized_columns[name.lower()]

    return None


def get_patient_column(df: pd.DataFrame) -> Optional[str]:
    return find_column(
        df,
        [
            "patient_id",
            "subject_id",
            "id",
            "patient",
        ],
    )


def get_medication_column(df: pd.DataFrame) -> Optional[str]:
    return find_column(
        df,
        [
            "medication",
            "medication_name",
            "drug",
            "drug_name",
            "description",
        ],
    )


def get_allergy_column(df: pd.DataFrame) -> Optional[str]:
    return find_column(
        df,
        [
            "allergy",
            "allergies",
            "allergy_name",
            "description",
        ],
    )


def empty_interaction_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "patient_id",
            "drug_a",
            "drug_b",
            "severity",
            "message",
        ]
    )


def empty_allergy_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "patient_id",
            "allergy",
            "medication",
            "severity",
            "message",
        ]
    )


def empty_duplicate_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "patient_id",
            "medication",
            "duplicate_count",
            "severity",
            "message",
        ]
    )


# ============================================================
# 4. DUPLICATE PRESCRIPTION DETECTION
# ============================================================

def detect_duplicate_prescriptions(
    medications_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Detect repeated medication entries for the same patient.

    Required columns:
    - patient_id or subject_id
    - medication, drug, or description
    """

    if medications_df is None or medications_df.empty:
        return empty_duplicate_dataframe()

    patient_col = get_patient_column(medications_df)
    medication_col = get_medication_column(medications_df)

    if not patient_col or not medication_col:
        raise ValueError(
            "Medication data must contain patient_id/subject_id "
            "and medication/drug/description columns."
        )

    df = medications_df.copy()

    df["_patient"] = df[patient_col].apply(normalize)
    df["_medication"] = df[medication_col].apply(normalize)

    df = df[
        (df["_patient"] != "")
        & (df["_medication"] != "")
    ]

    if df.empty:
        return empty_duplicate_dataframe()

    counts = (
        df.groupby(["_patient", "_medication"])
        .size()
        .reset_index(name="duplicate_count")
    )

    duplicates = counts[
        counts["duplicate_count"] > 1
    ].copy()

    if duplicates.empty:
        return empty_duplicate_dataframe()

    duplicates["severity"] = duplicates["duplicate_count"].apply(
        lambda count: "High" if count >= 3 else "Moderate"
    )

    duplicates["message"] = (
        "Same medication appears multiple times for the same patient."
    )

    duplicates.rename(
        columns={
            "_patient": "patient_id",
            "_medication": "medication",
        },
        inplace=True,
    )

    return duplicates[
        [
            "patient_id",
            "medication",
            "duplicate_count",
            "severity",
            "message",
        ]
    ].reset_index(drop=True)


# ============================================================
# 5. DRUG-DRUG INTERACTION DETECTION
# ============================================================

def detect_drug_interactions(
    medications_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Check medications belonging to the same patient
    against predefined interaction rules.
    """

    if medications_df is None or medications_df.empty:
        return empty_interaction_dataframe()

    patient_col = get_patient_column(medications_df)
    medication_col = get_medication_column(medications_df)

    if not patient_col or not medication_col:
        raise ValueError(
            "Medication data must contain patient_id/subject_id "
            "and medication/drug/description columns."
        )

    findings = []

    df = medications_df.copy()

    df["_patient"] = df[patient_col].apply(normalize)
    df["_medication"] = df[medication_col].apply(normalize)

    df = df[
        (df["_patient"] != "")
        & (df["_medication"] != "")
    ]

    for patient_id, group in df.groupby("_patient"):
        drugs = sorted(set(group["_medication"].tolist()))

        for index, drug_a in enumerate(drugs):
            for drug_b in drugs[index + 1:]:
                pair = frozenset({drug_a, drug_b})

                if pair in DRUG_INTERACTIONS:
                    rule = DRUG_INTERACTIONS[pair]

                    findings.append(
                        {
                            "patient_id": patient_id,
                            "drug_a": drug_a,
                            "drug_b": drug_b,
                            "severity": rule["severity"],
                            "message": rule["message"],
                        }
                    )

    if not findings:
        return empty_interaction_dataframe()

    return pd.DataFrame(findings).drop_duplicates(
        subset=[
            "patient_id",
            "drug_a",
            "drug_b",
        ]
    ).reset_index(drop=True)


# ============================================================
# 6. ALLERGY CONFLICT DETECTION
# ============================================================

def detect_allergy_conflicts(
    medications_df: pd.DataFrame,
    allergies_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compare each patient's known allergies with prescribed medicines.
    """

    if (
        medications_df is None
        or allergies_df is None
        or medications_df.empty
        or allergies_df.empty
    ):
        return empty_allergy_dataframe()

    medication_patient_col = get_patient_column(medications_df)
    medication_col = get_medication_column(medications_df)

    allergy_patient_col = get_patient_column(allergies_df)
    allergy_col = get_allergy_column(allergies_df)

    if not medication_patient_col or not medication_col:
        raise ValueError(
            "Invalid medication columns."
        )

    if not allergy_patient_col or not allergy_col:
        raise ValueError(
            "Invalid allergy columns."
        )

    medications = medications_df.copy()
    allergies = allergies_df.copy()

    medications["_patient"] = medications[
        medication_patient_col
    ].apply(normalize)

    medications["_medication"] = medications[
        medication_col
    ].apply(normalize)

    allergies["_patient"] = allergies[
        allergy_patient_col
    ].apply(normalize)

    allergies["_allergy"] = allergies[
        allergy_col
    ].apply(normalize)

    medications = medications[
        (medications["_patient"] != "")
        & (medications["_medication"] != "")
    ]

    allergies = allergies[
        (allergies["_patient"] != "")
        & (allergies["_allergy"] != "")
    ]

    findings = []

    for patient_id, medication_group in medications.groupby("_patient"):
        patient_allergies = allergies[
            allergies["_patient"] == patient_id
        ]

        if patient_allergies.empty:
            continue

        patient_medications = set(
            medication_group["_medication"].tolist()
        )

        patient_allergy_names = set(
            patient_allergies["_allergy"].tolist()
        )

        for allergy in patient_allergy_names:
            for medication in patient_medications:

                conflict = False

                # Direct allergy match
                if medication == allergy:
                    conflict = True

                # Category-based allergy match
                if allergy in ALLERGY_CONFLICTS:
                    if medication in ALLERGY_CONFLICTS[allergy]:
                        conflict = True

                if conflict:
                    findings.append(
                        {
                            "patient_id": patient_id,
                            "allergy": allergy,
                            "medication": medication,
                            "severity": "High",
                            "message": (
                                "Medication may conflict with known "
                                f"allergy: {allergy}."
                            ),
                        }
                    )

    if not findings:
        return empty_allergy_dataframe()

    return pd.DataFrame(findings).drop_duplicates(
        subset=[
            "patient_id",
            "allergy",
            "medication",
        ]
    ).reset_index(drop=True)


# ============================================================
# 7. COMPLETE PATIENT SAFETY CHECK
# ============================================================

def run_medication_safety_check(
    medications_df: pd.DataFrame,
    allergies_df: Optional[pd.DataFrame] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Run all medication safety checks.

    Returns:
        {
            "drug_interactions": DataFrame,
            "allergy_conflicts": DataFrame,
            "duplicate_prescriptions": DataFrame
        }
    """

    if medications_df is None:
        medications_df = pd.DataFrame()

    if allergies_df is None:
        allergies_df = pd.DataFrame()

    return {
        "drug_interactions": detect_drug_interactions(
            medications_df
        ),
        "allergy_conflicts": detect_allergy_conflicts(
            medications_df,
            allergies_df,
        ),
        "duplicate_prescriptions": detect_duplicate_prescriptions(
            medications_df
        ),
    }


# ============================================================
# 8. SAFETY SUMMARY
# ============================================================

def get_safety_summary(
    results: Dict[str, pd.DataFrame],
) -> Dict[str, int]:
    """
    Return simple counts for dashboard cards.
    """

    interactions = results.get(
        "drug_interactions",
        empty_interaction_dataframe(),
    )

    allergies = results.get(
        "allergy_conflicts",
        empty_allergy_dataframe(),
    )

    duplicates = results.get(
        "duplicate_prescriptions",
        empty_duplicate_dataframe(),
    )

    return {
        "drug_interactions": len(interactions),
        "allergy_conflicts": len(allergies),
        "duplicate_prescriptions": len(duplicates),
        "total_safety_findings": (
            len(interactions)
            + len(allergies)
            + len(duplicates)
        ),
    }


# ============================================================
# 9. COMMAND-LINE TEST
# ============================================================

if __name__ == "__main__":

    medications = pd.DataFrame(
        [
            {
                "patient_id": "P001",
                "medication": "warfarin",
            },
            {
                "patient_id": "P001",
                "medication": "aspirin",
            },
            {
                "patient_id": "P001",
                "medication": "warfarin",
            },
            {
                "patient_id": "P002",
                "medication": "amoxicillin",
            },
        ]
    )

    allergies = pd.DataFrame(
        [
            {
                "patient_id": "P002",
                "allergy": "penicillin",
            }
        ]
    )

    results = run_medication_safety_check(
        medications,
        allergies,
    )

    summary = get_safety_summary(results)

    print("\n=== DRUG-DRUG INTERACTIONS ===")
    print(
        results["drug_interactions"].to_string(index=False)
    )

    print("\n=== ALLERGY CONFLICTS ===")
    print(
        results["allergy_conflicts"].to_string(index=False)
    )

    print("\n=== DUPLICATE PRESCRIPTIONS ===")
    print(
        results["duplicate_prescriptions"].to_string(index=False)
    )

    print("\n=== SAFETY SUMMARY ===")
    print(summary)