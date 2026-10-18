from sqlalchemy import text
from database import engine


def check_duplicate_medications():
    """
    Same patient ko same medication multiple times prescribe hui hai ya nahi.
    """

    query = text("""
        SELECT
            "PATIENT",
            "DESCRIPTION" AS medication,
            COUNT(*) AS prescription_count
        FROM synthea_medications
        GROUP BY "PATIENT", "DESCRIPTION"
        HAVING COUNT(*) > 1
        ORDER BY prescription_count DESC;
    """)

    with engine.connect() as conn:
        rows = conn.execute(query).fetchall()

    print("\n========== DUPLICATE MEDICATIONS ==========")

    if not rows:
        print("Koi duplicate prescription nahi mili.")
    else:
        for row in rows:
            print(
                f"Patient: {row.PATIENT} | "
                f"Medication: {row.medication} | "
                f"Count: {row.prescription_count}"
            )


def check_allergy_conflicts():
    """
    Patient ki allergies aur prescribed medications ko cross-check karta hai.
    """

    query = text("""
        SELECT DISTINCT
            m."PATIENT",
            m."DESCRIPTION" AS medication,
            a."DESCRIPTION" AS allergy
        FROM synthea_medications m
        INNER JOIN synthea_allergies a
            ON m."PATIENT" = a."PATIENT"
        WHERE LOWER(m."DESCRIPTION") LIKE
              '%' || LOWER(a."DESCRIPTION") || '%'
           OR LOWER(a."DESCRIPTION") LIKE
              '%' || LOWER(m."DESCRIPTION") || '%';
    """)

    with engine.connect() as conn:
        rows = conn.execute(query).fetchall()

    print("\n========== ALLERGY CONFLICTS ==========")

    if not rows:
        print("Koi direct allergy conflict nahi mila.")
    else:
        for row in rows:
            print(
                f"Patient: {row.PATIENT} | "
                f"Medication: {row.medication} | "
                f"Allergy: {row.allergy}"
            )


def main():
    print("Medication Safety Checker Start ho raha hai...")

    check_duplicate_medications()
    check_allergy_conflicts()

    print("\nMedication Safety Check Complete!")


if __name__ == "__main__":
    main()