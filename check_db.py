from database import engine
from sqlalchemy import text

with engine.connect() as conn:
    db_name = conn.execute(text("SELECT current_database()")).scalar()
    addr = conn.execute(text("SELECT inet_server_addr()")).scalar()
    port = conn.execute(text("SELECT inet_server_port()")).scalar()

    print("App is connected to:")
    print("Database:", db_name)
    print("Server address:", addr)
    print("Port:", port)

    result = conn.execute(text("SELECT table_schema, table_name FROM information_schema.tables WHERE table_name = 'patient_risk_scores'"))

    rows = result.fetchall()

    if rows:
        print("Table MILI:")
        for row in rows:
            print(" - Schema:", row[0], "| Table:", row[1])
    else:
        print("Table NAHI MILI is database mein!")
