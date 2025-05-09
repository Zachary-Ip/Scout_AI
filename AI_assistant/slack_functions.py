import os
import sqlite3

import pandas as pd
from dotenv import find_dotenv, load_dotenv
from google.cloud import bigquery
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables.base import RunnableSequence
from langchain_openai import ChatOpenAI
from tabulate import tabulate

load_dotenv(find_dotenv())
OPENAI_API_KEY = os.getenv("OPENAI_KEY")

DEMOGRAPHIC_SCHEMA = """
# TABLE TO QUERY FROM: student_demographic
This database contains demographic and lifestyle data for students, including academic performance over three years. The table contains one row per student.

Column Name: Description (Data Type)

- sex: Student's biological sex (STRING;'M', 'F')
- age: Age in years (INTEGER)
- environment: Type of residence environment (STRING; 'Urban', 'Rural')
- famsize: Family size category (STRING; <3, >3)
- parent_living_situ: Living situation with parents (STRING; 'Together', 'Apart')

- Mother_edu: Mother's education level (INTEGER; 0–5 scale)
- Father_edu: Father's education level (INTEGER)
- Mjob: Mother's occupation (STRING;'teacher', 'services', 'health')
- Fjob: Father's occupation (STRING)
- guardian: Primary guardian (STRING; 'mother', 'father', 'other')

- traveltime: Commute time to school (INTEGER; 0–5 scale)
- studytime: Weekly study time (INTEGER)
- failures: Number of past academic failures (INTEGER)
- extra_classes: Enrolled in additional paid classes (BOOLEAN or STRING;'yes', 'no')
- family_support: Extra educational support from family (BOOLEAN or STRING)

- paid: Attending paid tutoring (BOOLEAN or STRING)
- activities: Participates in extracurriculars (BOOLEAN or STRING)
- nursery: Attended nursery school (BOOLEAN or STRING)
- internet: Has Internet access at home (BOOLEAN or STRING)
- romantic: Currently in a romantic relationship (BOOLEAN or STRING)

- family_quality: Quality of family relationships (INTEGER; 0–5 scale)
- freetime: Free time after school (INTEGER)
- goout: Frequency of going out with friends (INTEGER)
- Weekday_alchohol: Weekday alcohol consumption (INTEGER)
- Weekend_alchohol: Weekend alcohol consumption (INTEGER)
- health: Self-reported health status (INTEGER)
- num_absences: Number of school absences (INTEGER)

- Yr_1_Grade: Final grade for Year 1 (FLOAT; 0 - 1.0)
- Yr_2_Grade: Final grade for Year 2 (FLOAT)
- Yr_3_Grade: Final grade for Year 3 (FLOAT)
"""

DEMOGRAPHIC_QUERY = """
A query structured like this is preferred if relevant:
```
SELECT 
    <categorical_column>,
    ROUND(AVG(<continuous_column>), 2) AS <continuous_column>_Avg,
FROM 
    <table_name>
GROUP BY 
    <categorical_column>
ORDER BY 
    <categorical_column>;
```
Replace:
- <categorical_column> with the relevant categorical column from my data (e.g., Gender, Ethnicity)
- <continuous_column> with the relevant continuous variable to analyze (e.g., Yr_1_Grade, num_absences, traveltime)
- <table_name> with the appropriate table name
"""


def RAG_response(user_query, say):
    # Initialize chat model
    chat = ChatOpenAI(model_name="gpt-4o", temperature=1)

    # Long form replies
    debug = True if user_query.strip().startswith("DEBUG") else False

    # Prompt templates
    senior_prompt_template = PromptTemplate(
        input_variables=["user_query", "schema_info"],
        template="""
        # Role
        You are a senior data scientist helping to break down a user questions to understand what data would be needed to answer it.

        # Available Data
        
        {schema_info}
        
        # User Question
        "{user_query}"
        
        # Instructions
       
        1. Determine if the question is relevant to the available table.
        
        2. If the question CANNOT be answered with the available data, or if the question is too vague
        - Start your response with "RETURN " followed by a brief explanation
        - Example: "RETURN This question requires financial data not available in the schema."
        
        3. If the question CAN be answered with the available data start your response with CONTINUE
        
        4. Determine what underlying data is neccesary to answer a question
            Example: 
            - Q. Does alcohol negatively affect grades?
            - A. It sounds like you want to select columns related to alcohol use and grades, look at the average grade columns by grouping the alcohol columns
        - Reply only requesting to isolate the data that would be neccesary to answer the question
    """,
    )

    sql_prompt_template = PromptTemplate(
        input_variables=[
            "user_input",
            "schema_info",
            "query_structure",
        ],
        template="""
        # Instructions
         
        Write a syntactically valid sqlite3 query that answers the user's question using only the schema provided. 
        
        {query_structure}

        - The question might not use the exact same words or names as the columns, so use your best judgement to map the question to the schema provided.
        - Do not include any data modification statements (e.g., INSERT, UPDATE, DELETE).
        - Return only the SQL query, wrapped in ```sql``` tags — no additional text, explanation, or formatting.


        # Database Schema
        {schema_info}
        
        # User Query
        "{user_input}"    
        """,
    )

    final_prompt_template = PromptTemplate(
        input_variables=["user_query", "sql_result"],
        template="""
        # Context
        You are a senior data scientist. A junior data scientist has performed an SQL query and given you the results. Use the information provided to synthesis an answer to the user question. 
        - Do not use external knowledge, but you may assume that you have been provided enough information to answer the question, even if the table does not explicitly state it. 
        - If the answer is not clearly supported by the data, say so.
        Example:

        {sql_result}

        # User Question
        {user_query}
        """,
    )

    # Initialize chains

    senior_chain = senior_prompt_template | chat
    sql_generation_chain = sql_prompt_template | chat
    final_response_chain = final_prompt_template | chat

    # Run the SQL generation chain
    senior_response = senior_chain.invoke(
        input={
            "user_query": user_query,
            "schema_info": DEMOGRAPHIC_SCHEMA,
        }
    )
    senior_text = senior_response.text().strip()
    if senior_text.startswith("RETURN"):
        return senior_text.strip("RETURN")
    elif senior_text.startswith("CONTINUE"):
        senior_text = senior_text.strip("CONTINUE").strip()
        sql_response = sql_generation_chain.invoke(
            input={
                "user_input": user_query,
                "schema_info": DEMOGRAPHIC_SCHEMA,
                "query_structure": DEMOGRAPHIC_QUERY,
            }
        )
        if debug:
            say(senior_text)
        # Extract the SQL query
        SQL_query = extract_sql_from_response(sql_response.text())
        if debug:
            say("Querying the database with:")
            say(SQL_query)

        # Execute the SQL query
        conn = sqlite3.connect("student.db")
        try:
            df = pd.read_sql_query(SQL_query, conn)
            text_table = df.to_markdown(index=False)
            say("Here's what I found:")
            say(f"```{text_table}```")
        except Exception as e:
            say("`Error retrieving data, invalid query`")
            print(e)

    else:
        return f"`ERROR, template not strict enough.` Chat responded:\n``` {senior_response.text()}\n```"

    # Run the final response chain
    return final_response_chain.invoke(
        input={"user_query": user_query, "sql_result": text_table}
    ).text()


def extract_sql_from_response(llm_response):
    # Look for SQL code blocks
    import re

    sql_pattern = r"```sql\s*(.*?)\s*```"
    matches = re.findall(sql_pattern, llm_response, re.DOTALL)

    if matches:
        return matches[0].strip()
    else:
        # Fallback if no code blocks found
        return llm_response.strip()


def execute_bigquery_sql(sql_query, timeout=30):
    client = bigquery.Client()

    try:
        # Execute the query
        query_job = client.query(sql_query, timeout=timeout)

        # Convert results to a list of dictionaries
        results = [dict(row) for row in query_job]
        formatted = tabulate(results, headers="keys", tablefmt="github")
        return formatted, None
    except Exception as e:
        return None, str(e)
