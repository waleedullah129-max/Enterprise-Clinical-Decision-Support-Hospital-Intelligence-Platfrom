"""
Clinical Knowledge Graph Builder
=================================
Ye script Postgres (Synthea data) se Patient, Condition aur Encounter
data nikal kar Neo4j mein ek connected knowledge graph banati hai.

Graph structure:
    (Patient) -[:HAS_CONDITION]-> (Condition)
    (Patient) -[:HAD_ENCOUNTER]-> (Encounter)
    (Encounter) -[:DIAGNOSED]-> (Condition)   [agar encounter/condition link ho]

Pehle install karein (agar nahi kiya):
    pip install neo4j sqlalchemy psycopg2-binary
"""

import traceback
from sqlalchemy import text
from neo4j import GraphDatabase
from database import engine   # aapki maujooda database.py se Postgres engine

# =========================
# CONFIG
# =========================

# neo4j+ssc use kar rahe hain kyunke aapke system pe SSL cert verification
# issue tha (antivirus/network ki wajah se) - agar us system pe kaam na kare
# to "neo4j+s" try karein.
NEO4J_URI = "neo4j+ssc://38f667be.databases.neo4j.io"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "_PBeNvDYknPHwy6I9KrWiVrirf0aLWbGm8WsxH0mWJ4"

BATCH_SIZE = 500


# =========================
# STEP 1: Postgres se data nikalna
# =========================

def fetch_patients():
    query = text("""
        SELECT "Id", "FIRST", "LAST", "GENDER", "BIRTHDATE"
        FROM synthea_patients
    """)
    with engine.connect() as conn:
        rows = [dict(r._mapping) for r in conn.execute(query)]
    print(f"Patients mile: {len(rows)}")
    return rows


def fetch_conditions():
    # NOTE: agar aapke table/column names alag hain, to yahan update karein
    query = text("""
        SELECT "PATIENT", "CODE", "DESCRIPTION", "START", "STOP"
        FROM synthea_conditions
    """)
    with engine.connect() as conn:
        rows = [dict(r._mapping) for r in conn.execute(query)]
    print(f"Conditions mile: {len(rows)}")
    return rows


def fetch_encounters():
    # NOTE: agar aapke table/column names alag hain, to yahan update karein
    query = text("""
        SELECT "Id", "PATIENT", "ENCOUNTERCLASS", "START", "STOP", "REASONDESCRIPTION"
        FROM synthea_encounters
    """)
    with engine.connect() as conn:
        rows = [dict(r._mapping) for r in conn.execute(query)]
    print(f"Encounters mile: {len(rows)}")
    return rows


# =========================
# STEP 2: Neo4j mein nodes aur relationships banana
# =========================

def push_patients(session, patients):
    query = """
    UNWIND $rows AS row
    MERGE (p:Patient {id: row.Id})
    SET p.first_name = row.FIRST,
        p.last_name = row.LAST,
        p.gender = row.GENDER,
        p.birthdate = row.BIRTHDATE
    """
    for i in range(0, len(patients), BATCH_SIZE):
        batch = patients[i:i + BATCH_SIZE]
        session.run(query, rows=batch)
        print(f"  Patient batch {i // BATCH_SIZE + 1} done ({len(batch)} rows).")


def push_conditions(session, conditions):
    query = """
    UNWIND $rows AS row
    MERGE (c:Condition {code: row.CODE})
    SET c.description = row.DESCRIPTION
    WITH row, c
    MATCH (p:Patient {id: row.PATIENT})
    MERGE (p)-[r:HAS_CONDITION]->(c)
    SET r.start_date = row.START,
        r.stop_date = row.STOP
    """
    for i in range(0, len(conditions), BATCH_SIZE):
        batch = conditions[i:i + BATCH_SIZE]
        session.run(query, rows=batch)
        print(f"  Condition batch {i // BATCH_SIZE + 1} done ({len(batch)} rows).")


def push_encounters(session, encounters):
    query = """
    UNWIND $rows AS row
    MERGE (e:Encounter {id: row.Id})
    SET e.encounter_class = row.ENCOUNTERCLASS,
        e.start_date = row.START,
        e.stop_date = row.STOP,
        e.reason = row.REASONDESCRIPTION
    WITH row, e
    MATCH (p:Patient {id: row.PATIENT})
    MERGE (p)-[:HAD_ENCOUNTER]->(e)
    """
    for i in range(0, len(encounters), BATCH_SIZE):
        batch = encounters[i:i + BATCH_SIZE]
        session.run(query, rows=batch)
        print(f"  Encounter batch {i // BATCH_SIZE + 1} done ({len(batch)} rows).")


# =========================
# MAIN
# =========================

def main():
    # Postgres se data nikalna
    patients = fetch_patients()
    try:
        conditions = fetch_conditions()
    except Exception:
        print("synthea_conditions table nahi mili ya error aayi - is step ko skip kar rahe hain.")
        conditions = []

    try:
        encounters = fetch_encounters()
    except Exception:
        print("synthea_encounters table nahi mili ya error aayi - is step ko skip kar rahe hain.")
        encounters = []

    if not patients:
        print("Koi patient data nahi mila. Postgres connection/table check karein.")
        return

    # Neo4j mein push karna
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    try:
        with driver.session() as session:
            print("\nPatients Neo4j mein daal rahe hain...")
            push_patients(session, patients)

            if conditions:
                print("\nConditions Neo4j mein daal rahe hain aur Patient se link kar rahe hain...")
                push_conditions(session, conditions)

            if encounters:
                print("\nEncounters Neo4j mein daal rahe hain aur Patient se link kar rahe hain...")
                push_encounters(session, encounters)

        print("\nKnowledge graph safaltapoorvak ban gaya!")
    except Exception:
        print("Neo4j mein data daalte waqt error aayi:")
        traceback.print_exc()
    finally:
        driver.close()


if __name__ == "__main__":
    main()