import pandas as pd
from sqlalchemy import create_engine
from urllib.parse import quote_plus

# ============================================
# 1. DATABASE CONNECTION (PostgreSQL)
# ============================================
DB_USER = "postgres"
DB_PASSWORD = "wale@123"
DB_HOST = "localhost"
DB_PORT = "5432"          # Postgres ka default port
DB_NAME = "my_database"

safe_password = quote_plus(DB_PASSWORD)

engine = create_engine(
    f"postgresql+psycopg2://{DB_USER}:{safe_password}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

print("Database se connect ho raha hai...")

# ============================================
# 2. MIMIC-IV DATA LOAD KARO
# ============================================
print("\n--- MIMIC-IV data load ho raha hai ---")

mimic_patients = pd.read_csv('data/mimic/hosp/patients.csv.gz')
mimic_patients.to_sql('mimic_patients', engine, if_exists='replace', index=False)
print(f"mimic_patients table banaya: {mimic_patients.shape[0]} rows")

mimic_admissions = pd.read_csv('data/mimic/hosp/admissions.csv.gz')
mimic_admissions.to_sql('mimic_admissions', engine, if_exists='replace', index=False)
print(f"mimic_admissions table banaya: {mimic_admissions.shape[0]} rows")

mimic_diagnoses = pd.read_csv('data/mimic/hosp/diagnoses_icd.csv.gz')
mimic_diagnoses.to_sql('mimic_diagnoses', engine, if_exists='replace', index=False)
print(f"mimic_diagnoses table banaya: {mimic_diagnoses.shape[0]} rows")

mimic_labs = pd.read_csv('data/mimic/hosp/labevents.csv.gz')
mimic_labs.to_sql('mimic_labevents', engine, if_exists='replace', index=False, chunksize=5000)
print(f"mimic_labevents table banaya: {mimic_labs.shape[0]} rows")

mimic_icustays = pd.read_csv('data/mimic/icu/icustays.csv.gz')
mimic_icustays.to_sql('mimic_icustays', engine, if_exists='replace', index=False)
print(f"mimic_icustays table banaya: {mimic_icustays.shape[0]} rows")

# ============================================
# 3. SYNTHEA DATA LOAD KARO
# ============================================
print("\n--- Synthea data load ho raha hai ---")

synthea_patients = pd.read_csv('data/synthea/patients.csv')
synthea_patients.to_sql('synthea_patients', engine, if_exists='replace', index=False)
print(f"synthea_patients table banaya: {synthea_patients.shape[0]} rows")

synthea_conditions = pd.read_csv('data/synthea/conditions.csv')
synthea_conditions.to_sql('synthea_conditions', engine, if_exists='replace', index=False)
print(f"synthea_conditions table banaya: {synthea_conditions.shape[0]} rows")

synthea_medications = pd.read_csv('data/synthea/medications.csv')
synthea_medications.to_sql('synthea_medications', engine, if_exists='replace', index=False)
print(f"synthea_medications table banaya: {synthea_medications.shape[0]} rows")

synthea_encounters = pd.read_csv('data/synthea/encounters.csv')
synthea_encounters.to_sql('synthea_encounters', engine, if_exists='replace', index=False)
print(f"synthea_encounters table banaya: {synthea_encounters.shape[0]} rows")

synthea_allergies = pd.read_csv('data/synthea/allergies.csv')
synthea_allergies.to_sql('synthea_allergies', engine, if_exists='replace', index=False)
print(f"synthea_allergies table banaya: {synthea_allergies.shape[0]} rows")

print("\n✅ Sab data successfully PostgreSQL database mein load ho gaya!")