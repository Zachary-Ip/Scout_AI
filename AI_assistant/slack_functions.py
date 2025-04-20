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
SCHEMA_INFO = """
Table: bigquery-public-data.google_trends.top_terms
Columns:
- dma_name (STRING): The name of the designated market area
- term (STRING): The search term
- week (DATE): The week of the trend data
- score (INTEGER): The popularity score of the term
- rank (INTEGER): The rank of the term for that week and DMA
"""

SURVEY_SCHEMA = """

# TABLE TO QUERY FROM: student_survey

This table contains survey results where each column corresponds to a student self-assessment question.
All responses are numeric, ranging from 0 to 4:
    0 = Never / Very Poor
    1 = Rarely / Poor
    2 = Sometimes / Average
    3 = Most of the time / Good
    4 = Always / Excellent

Each column is listed below as: [Column Name]: [Question]

GPA: Student's  (on a 4-point scale)
Q1: Do you make time for exercise and socializing?
Q2: Do you get at least 6 hours of sleep per night?
Q3: Do you study at least 2 hours for every hour of class?
Q4: Do you have a consistent study location?
Q5: Is your study area quiet, comfortable, and distraction-free?
Q6: Can you study for 30+ minutes without breaks?
Q7: Do you use time between classes for studying?
Q8: Do you begin reviewing for major exams 3+ days in advance?
Q9: Do you know what kinds of questions will be on tests?
Q10: Are you able to finish tests in the allowed time?
Q11: Do you complete assignments without using solution guides?
Q12: Do you ask questions in class when you're confused?
Q13: Can you take notes, keep up, and understand during lectures?
Q14: Do you review your notes shortly after class?
Q15: Do you annotate/highlight class materials while reading?
Q16: Can you read 12–15 pages/hour for history-type material?
Q17: Can you understand readings without needing to re-read?
Q18: Do you adjust your reading style for different subjects?
"""

SURVEY_QUERY = """
A query structured like this is preferred if relevant:
```
SELECT
    <Q_x> AS <x_label>,
    COUNT(*) AS Student_Count,
    ROUND(AVG(<Q_y>), 2) AS Avg_<y_label>,
    ROUND(AVG(GPA), 2) AS Avg_GPA
FROM
    student_survey
GROUP BY
    <Q_x>
ORDER BY
    <Q_x>;
```
Expand the number of labels as neccessary for relevant questions.
"""

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
    ROUND(AVG(<continuous_column>), 2) AS Avg,
    ROUND(SQRT(AVG(<continuous_column>*<continuous_column>) - AVG(<continuous_column>)*AVG(<continuous_column>)), 2) AS Std
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
        input_variables=["user_query"],
        template="""
        # Role
        You are a senior data scientist helping to translate user questions for a junior data scientist to create an SQL query for.
        
        # Available Data
        You have access to 2 data sets:
            1. student survey data - a survey of student study habits on a scale from 1 (never) to 5 (always), and GPA. Questions cover topics like: 
            - "did you sleep 6 hours"
            - "do you ask questions in class"
            - "Do you review notes"
            - current GPA (no year specified)

            2. student demographic data - a table containing demographic data grades across 3 years. Contains information such as
            - Age of student
            - environment (rural vs urban)
            - parents education (ranks 1-5), status (together, separated)
            - self reported free time, study time, family relationship, alcohol usage
            - absences and grades for years 1 through 3
        
        # User Question
        "{user_query}"
        
        # Instructions
       
        1. Determine if the question is relevant to the available tables.
        
        2. If the question CANNOT be answered with the available data, or if the question is too vague
        - Start your response with "RETURN " followed by a brief explanation
        - Example: "RETURN This question requires financial data not available in the schema."
        
        3. If the question CAN be answered with the available data:
        - Start your response with "CONTINUE" then either "SURVEY" or "DEMOGRAPHIC", depending on which table is more appropriate for the question 

        4. Finally, Determine what underlying data is neccesary to answer a question
            Example: 
            - Q. Does alcohol negatively affect grades?
            - A. It sounds like you want to select columns related to alcohol use and grades, look at the average grade columns by grouping the alcohol columns
        - Rephrase the question to isolate data from the database that is relevant to the question
        - The query should be concise and include only essential columns
        - Limit results to a reasonable number if returning raw records
        - Do not use INSERT, UPDATE, or DELETE statements
    """,
    )

    sql_prompt_template = PromptTemplate(
        input_variables=["user_input", "schema_info", "query_structure"],
        template="""
        # Role
        You are a senior data scientist providing insight to a question by creating an SQL query for a database.

        # Instructions
         
        Write a syntactically valid BigQuery Standard SQL query that answers the user's question using only the schema provided. 
        
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
    senior_response = senior_chain.invoke(input={"user_query": user_query})
    senior_text = senior_response.text().strip()
    if senior_text.startswith("RETURN"):
        return senior_text.strip("RETURN")
    elif senior_text.startswith("CONTINUE"):
        senior_text = senior_text.strip("CONTINUE").strip()
        if senior_text.startswith("SURVEY"):
            senior_text = senior_text.strip("SURVEY").strip()
            if debug:
                say("That seems like a good question for the Survey table")
            # Run the SQL generation chain
            sql_response = sql_generation_chain.invoke(
                input={
                    "user_input": senior_text,
                    "schema_info": SURVEY_SCHEMA,
                    "query_structure": SURVEY_QUERY,
                }
            )
        elif senior_text.startswith("DEMOGRAPHIC"):
            if debug:
                say("That seems like a good question for the Demographic table")
            senior_text = senior_text.strip("DEMOGRAPHIC").strip()
            sql_response = sql_generation_chain.invoke(
                input={
                    "user_input": senior_text,
                    "schema_info": DEMOGRAPHIC_SCHEMA,
                    "query_structure": DEMOGRAPHIC_QUERY,
                }
            )
        else:
            return f"`ERROR: I was not able to route you to an appropriate table.` \n Response: \n ```{senior_text}```"
        if debug:
            say(f"Let's try phrasing your question like this: {senior_text}")
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
