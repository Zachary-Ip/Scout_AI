import pandas as pd
import sqlite3

# Define your files and table names
csv_files = {
    "student_survey": "data/in1.csv",
    "student_demographic": "data/student_data.csv",
}

# Create (or connect to) SQLite database
conn = sqlite3.connect("student.db")

# Loop through files and load them
for table_name, file_path in csv_files.items():
    df = pd.read_csv(file_path)
    df.to_sql(table_name, conn, if_exists="replace", index=False)
    print(f"Loaded {file_path} into table '{table_name}'")

conn.close()
