"""
alert_center.py — Hospital Alert Center
==========================================
Poore dataset ko scan karta hai aur real-time alerts generate karta hai:
    - High-Risk Patient alerts (ML risk model se)
    - Drug Interaction alerts
    - Allergy Conflict alerts
    - Duplicate Prescription alerts
    - ICU Capacity alerts (agar utilization zyada ho)

Ye FastAPI backend (main.py) se data leta hai — seedha database access nahi
karta, isi consistent architecture ko follow karta hai jo dashboard.py aur
copilot.py mein hai.

Chalane ka tarika (standalone test):
    python alert_center.py

FastAPI mein already GET /alerts endpoint bhi add ho chuka hai (main.py mein) —
wo isi module ke generate_alerts() function ko reuse karta hai.
"""

from __future__ import annotations

from typing import Any, Dict, List

import requests


API_BASE_URL = "http://127.0.0.1:8000"

# Agar ICU stays is number se zyada hon, capacity alert generate hoga
# (demo threshold — real hospital mein ye bed-count se calculate hoga)
ICU_CAPACITY_ALERT_THRESHOLD = 100


# ============================================
# HELPER: FastAPI se data lao
# ============================================
def _get(path: str, params: dict | None = None):
    response = requests.get(f"{API_BASE_URL}{path}", params=params, timeout=30)
    response.raise_for_status()
    return response.json()


# ============================================
# ALERT GENERATORS (har ek ek category)
# ============================================
def get_high_risk_patient_alerts():
    scored = _get("/risk-scores")
    alerts = []

    # FastAPI response agar dictionary mein list bhej raha ho
    if isinstance(scored, dict):
        patients = (
            scored.get("risk_scores")
            or scored.get("results")
            or scored.get("data")
            or []
        )
    else:
        patients = scored

    for patient in patients:
        if not isinstance(patient, dict):
            continue

        risk_label = patient.get("risk_label") or patient.get("risk_level")

        if risk_label == "High Risk":
            alerts.append({
                "type": "High Risk Patient",
                "patient_id": patient.get("subject_id"),
                "message": (
                    f"Patient {patient.get('subject_id')} is at high risk."
                ),
                "risk_score": patient.get("risk_score")
            })

    return alerts


def get_drug_interaction_alerts() -> List[Dict[str, Any]]:
    """Sab patients mein drug-drug interactions ke liye alerts."""
    findings = _get("/medication-safety/drug-interactions")
    alerts = []

    for finding in findings:
        alerts.append(
            {
                "category": "Drug Interaction",
                "severity": finding["severity"],
                "patient_id": finding["patient_id"],
                "message": (
                    f"{finding['drug_a']} + {finding['drug_b']} — {finding['message']}"
                ),
            }
        )

    return alerts


def get_allergy_conflict_alerts() -> List[Dict[str, Any]]:
    """Sab patients mein allergy conflicts ke liye alerts."""
    findings = _get("/medication-safety/allergy-conflicts")
    alerts = []

    for finding in findings:
        alerts.append(
            {
                "category": "Allergy Conflict",
                "severity": finding["severity"],
                "patient_id": finding["patient_id"],
                "message": (
                    f"Medication '{finding['medication']}' documented allergy "
                    f"'{finding['allergy']}' se conflict karti hai."
                ),
            }
        )

    return alerts


def get_duplicate_prescription_alerts() -> List[Dict[str, Any]]:
    """High-severity duplicate prescriptions ke liye alerts (Moderate skip karte hain, warna bohot zyada aa jayenge)."""
    findings = _get("/medication-safety/duplicate-prescriptions")
    alerts = []

    for finding in findings:
        if finding["severity"] == "High":
            alerts.append(
                {
                    "category": "Duplicate Prescription",
                    "severity": finding["severity"],
                    "patient_id": finding["patient_id"],
                    "message": (
                        f"'{finding['medication']}' {finding['duplicate_count']} baar "
                        f"prescribe hui — pharmacist review chahiye."
                    ),
                }
            )

    return alerts


def get_icu_capacity_alert() -> List[Dict[str, Any]]:
    """Agar ICU utilization threshold se zyada ho to ek hospital-wide alert."""
    icu = _get("/icu/utilization")
    total_stays = icu.get("total_icu_stays", 0)

    if total_stays >= ICU_CAPACITY_ALERT_THRESHOLD:
        return [
            {
                "category": "ICU Capacity",
                "severity": "High",
                "patient_id": None,
                "message": (
                    f"ICU utilization high hai: {total_stays} total stays "
                    f"(threshold {ICU_CAPACITY_ALERT_THRESHOLD}). Bed allocation review karo."
                ),
            }
        ]

    return []


# ============================================
# SAB ALERTS EK SAATH
# ============================================
def generate_alerts() -> Dict[str, Any]:
    """
    Sab alert categories ko combine karta hai aur severity ke hisaab se
    sort kar ke return karta hai (High pehle).
    """
    all_alerts: List[Dict[str, Any]] = []
    all_alerts += get_high_risk_patient_alerts()
    all_alerts += get_drug_interaction_alerts()
    all_alerts += get_allergy_conflict_alerts()
    all_alerts += get_duplicate_prescription_alerts()
    all_alerts += get_icu_capacity_alert()

    severity_order = {"High": 0, "Moderate": 1, "Low": 2}
    all_alerts.sort(key=lambda a: severity_order.get(a["severity"], 3))

    summary = {
        "total_alerts": len(all_alerts),
        "high_severity": sum(1 for a in all_alerts if a["severity"] == "High"),
        "moderate_severity": sum(1 for a in all_alerts if a["severity"] == "Moderate"),
    }

    return {"summary": summary, "alerts": all_alerts}


# ============================================
# COMMAND-LINE TEST
# ============================================
if __name__ == "__main__":
    print("=== Hospital Alert Center ===\n")
    result = generate_alerts()

    print(f"Total alerts: {result['summary']['total_alerts']}")
    print(f"High severity: {result['summary']['high_severity']}")
    print(f"Moderate severity: {result['summary']['moderate_severity']}\n")

    for alert in result["alerts"][:30]:
        patient = alert["patient_id"] or "Hospital-wide"
        print(f"[{alert['severity']}] {alert['category']} | {patient}")
        print(f"   {alert['message']}\n")