from sqlalchemy import create_engine
from urllib.parse import quote_plus

DB_USER = "postgres"
DB_HOST = "localhost"
DB_PORT = "5432"
DB_NAME = "my_database"
DB_PASSWORD = "wale@123"   # <-- ye line update karein

safe_password = quote_plus(DB_PASSWORD)

engine = create_engine(
    f"postgresql+psycopg2://{DB_USER}:{safe_password}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

print("Database se connect ho raha hai...")