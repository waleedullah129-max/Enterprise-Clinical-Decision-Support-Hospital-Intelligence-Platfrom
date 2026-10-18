import traceback
from neo4j import GraphDatabase

NEO4J_URI = "neo4j+ssc://38f667be.databases.neo4j.io"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "_PBeNvDYknPHwy6I9KrWiVrirf0aLWbGm8WsxH0mWJ4"

try:
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    driver.verify_connectivity()
    print("Connection successful! Neo4j se connect ho gaya.")
    driver.close()
except Exception as e:
    print("Connection FAIL hua.")
    print(f"Error type: {type(e).__name__}")
    print(f"Error details: {e}")
    print("---- FULL TRACEBACK ----")
    traceback.print_exc()
