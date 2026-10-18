"""
Postgres (synthea_patients) se Neo4j Aura mein patient data migrate karne ka script.

Pehle ye install karein:
    pip install neo4j sqlalchemy psycopg2-binary

Phir neeche di gayi CONFIG values apni details se replace karein.
"""

from sqlalchemy import text
from neo4j import GraphDatabase
from database import engine   # aapki maujooda database.py se engine import ho raha hai

# =========================
# CONFIG - sirf Neo4j details yahan chahiye
# =========================

# Neo4j Aura connection (Query page se "Connection URI" copy karein)
NEO4J_URI = "neo4j+s://38f667be.databases.neo4j.io"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "_PBeNvDYknPHwy6I9KrWiVrirf0aLWbGm8WsxH0mWJ4"

# Kitne records ek batch mein bhejne hain (bohat bada data ho to ye useful hai)
BATCH_SIZE = 500


# =========================
# STEP 1: Postgres se data nikalna
# =========================

def fetch_patients_from_postgres():
    query = text("""
        SELECT "Id", "FIRST", "LAST", "GENDER", "BIRTHDATE"
        FROM synthea_patients
    """)

    with engine.connect() as conn:
        result = conn.execute(query)
        rows = [dict(row._mapping) for row in result]

    print(f"Postgres se {len(rows)} patients mile.")
    return rows


# =========================
# STEP 2: Neo4j mein daalna
# =========================

def push_patients_to_neo4j(patients):
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    # Cypher query jo ek batch of patients ko nodes bana degi
    # MERGE isliye use kar rahe hain taake duplicate na banein agar dobara chalayein
    cypher_query = """
    UNWIND $patients AS patient
    MERGE (p:Patient {id: patient.Id})
    SET p.first_name = patient.FIRST,
        p.last_name = patient.LAST,
        p.gender = patient.GENDER,
        p.birthdate = patient.BIRTHDATE
    """

    with driver.session() as session:
        # Data ko chhote batches mein todna (bohat zyada records ek saath bhejna slow ho sakta hai)
        for i in range(0, len(patients), BATCH_SIZE):
            batch = patients[i:i + BATCH_SIZE]
            session.run(cypher_query, patients=batch)
            print(f"Batch {i // BATCH_SIZE + 1} imported ({len(batch)} patients).")

    driver.close()
    print("Neo4j mein saara data safaltapoorvak import ho gaya! ✅")


# =========================
# MAIN
# =========================

if __name__ == "__main__":
    patients = fetch_patients_from_postgres()

    if patients:
        push_patients_to_neo4j(patients)
    else:
        print("Postgres se koi patient data nahi mila. Table/connection check karein.")