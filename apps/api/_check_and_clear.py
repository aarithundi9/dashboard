import os
from dotenv import load_dotenv
load_dotenv('c:/Users/aarit/OneDrive/Documents/GitHub/dashboard/apps/api/.env')
from databricks import sql

conn = sql.connect(
    server_hostname=os.environ['DATABRICKS_SERVER_HOSTNAME'],
    http_path=os.environ['DATABRICKS_HTTP_PATH'],
    access_token=os.environ['DATABRICKS_TOKEN'],
)
cur = conn.cursor()

catalog = os.getenv('DATABRICKS_CATALOG', 'workspace')
schema  = os.getenv('DATABRICKS_SCHEMA', 'default')

cur.execute(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{schema}")
cur.execute(f"DROP TABLE IF EXISTS {catalog}.{schema}.agent_runs")
cur.execute(f"DROP TABLE IF EXISTS {catalog}.{schema}.agent_steps")
print(f"Cleared {catalog}.{schema} — ready for fresh seed.")

cur.close()
conn.close()
