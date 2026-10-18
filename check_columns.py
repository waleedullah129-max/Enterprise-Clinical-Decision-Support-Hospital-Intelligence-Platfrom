from database import engine
from sqlalchemy import inspect

inspector = inspect(engine)

print("MEDICATIONS COLUMNS:")
print([c["name"] for c in inspector.get_columns("synthea_medications")])

print("\nALLERGIES COLUMNS:")
print([c["name"] for c in inspector.get_columns("synthea_allergies")])