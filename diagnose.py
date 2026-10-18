import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from database import engine
from sqlalchemy import text

# Get one real patient ID
with engine.connect() as conn:
    row = conn.execute(text('SELECT "Id" FROM synthea_patients LIMIT 1')).fetchone()
    patient_id = row[0]

print(f"Testing with patient_id = {patient_id}\n")

print("=== Testing predict_risk_for_patient() ===")
try:
    from risk_engine import predict_risk_for_patient
    result = predict_risk_for_patient(patient_id)
    print("SUCCESS:", result)
except Exception as e:
    import traceback
    print("FAILED with error:")
    traceback.print_exc()

print("\n=== Testing run_medication_safety_check() ===")
try:
    import pandas as pd
    from medication_safety import run_medication_safety_check

    meds = pd.read_sql(
        text('SELECT * FROM synthea_medications WHERE "PATIENT" = :pid'),
        conn if False else engine.connect(),
        params={"pid": patient_id}
    )
    meds.columns = [c.upper() for c in meds.columns]

    allergies = pd.read_sql(
        text('SELECT * FROM synthea_allergies LIMIT 500'),
        engine.connect()
    )
    allergies.columns = [c.upper() for c in allergies.columns]

    result = run_medication_safety_check(medications=meds, allergies=allergies)
    print("SUCCESS:", result)
except Exception as e:
    import traceback
    print("FAILED with error:")
    traceback.print_exc()