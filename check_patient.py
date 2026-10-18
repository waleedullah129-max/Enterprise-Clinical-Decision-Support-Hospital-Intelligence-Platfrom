from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from database import engine

def get_patient(patient_id: str):
    query = text("""
        SELECT "Id", "FIRST", "LAST", "GENDER", "BIRTHDATE"
        FROM synthea_patients
        WHERE "Id" = :patient_id
    """)

    try:
        with engine.connect() as conn:
            result = conn.execute(query, {"patient_id": patient_id})
            patient = result.fetchone()

        if patient:
            patient_dict = dict(patient._mapping)
            print("Patient ID valid hai.")
            print(patient_dict)
            return patient_dict
        else:
            print("Patient ID database mein nahi mili.")
            return None

    except SQLAlchemyError as e:
        print(f"Database error aayi: {e}")
        return None


if __name__ == "__main__":
    patient_id = "1dabc422-e007-d5ee-791f-9e08a8fd9b6a"
    get_patient(patient_id)