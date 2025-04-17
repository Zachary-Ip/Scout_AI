import os

from dotenv import load_dotenv
from google.cloud import bigquery

load_dotenv()  # Load from .env
print(os.getenv("GOOGLE_APPLICATION_CREDENTIALS"))
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.getenv(
    "GOOGLE_APPLICATION_CREDENTIALS"
)
bg_client = bigquery.Client()

query_text = """
SELECT
  term,
  ROUND(AVG(rank), 2) AS avg_rank,
  COUNT(*) AS num_entries
FROM `bigquery-public-data.google_trends.top_terms`
GROUP BY term
ORDER BY avg_rank ASC
LIMIT 10;
"""

q_job = bg_client.query(query_text)

print(q_job)
for row in q_job.result():
    print(row)
