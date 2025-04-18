import os

from dotenv import find_dotenv, load_dotenv
from google.cloud import bigquery
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_community.chat_models import ChatOpenAI
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


def RAG_response(user_input, say):
    # Initialize chat model
    chat = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=1)

    # 1. Routing and massaging query
    senior_prompt_template = PromptTemplate(
        input_variables=["user_input"],
        template="""
        # Role
        You are a senior data scientist helping to translate user questions for a junior data scientist to create an SQL query for.
        
        # Available Data
        You have access to 2 data sets:
            1. student survey data - a qualtrics survey of student study habits on a scale from 1 (never) to 5 (always), and GPA
            2. student demographic data - a table containing demographic data (e.g. age, gender, parents education), and grades in 3 classes
        
        # User Question
        "{user_query}"
        
        # Instructions
        1. Determine if the question can be answered using the available data.
        
        2. If the question CANNOT be answered with the available data, or if it's too vague:
        - Start your response with "RETURN: " followed by a brief explanation
        - Example: "RETURN: This question requires financial data not available in the schema."
        
        3. If the question CAN be answered with the available data:
        - Start your response with "CONTINUE: " followed by either "SURVEY: " or "DEMOGRAPHIC: ", then finally, a clear, rephrased version of the question
        - The junior data scientist will have access to the full data schema
        - The query should be concise and include only essential columns
        - Use GROUP BY, aggregation functions (AVG, COUNT, etc.) when appropriate for summarization
        - Include proper WHERE clauses to filter irrelevant data
        - Limit results to a reasonable number if returning raw records
        - Do not use INSERT, UPDATE, or DELETE statements
    """,
    )

    senior_chain = LLMChain(llm=chat, prompt=senior_prompt_template)

    # Run the SQL generation chain
    senior_response = senior_chain.run(user_input=user_input)

    if senior_response.startswith("RETURN: "):
        senior_response.strip("RETURN: ")
        say(senior_response)

    # 1. SQL Generation Chain
    sql_prompt_template = PromptTemplate(
        input_variables=["user_input", "schema_info"],
        template="""
        # Instructions
         
        Write a syntactically valid BigQuery Standard SQL query that answers the user's question using only the schema provided. 
        Do not include any data modification statements (e.g., INSERT, UPDATE, DELETE).
        Return only the SQL query, wrapped in ```sql``` tags — no additional text, explanation, or formatting.

        The data contains individual ranks of search terms for each week, and DMA, so querys that ask for 
        "highest ranking terms" without other specification will need to average across these terms to retrieve sensible values

        Also, it may be neccesary to re-rank  the outputs based on the score paremeter due to averageing across week or DMA area.

        # Database Schema
        {schema_info}
        
        # User Query
        "{user_input}"

        # Result limit
        Always limit results to not overwhelm the context window at
        LIMIT < 50;

        
        """,
    )

    sql_generation_chain = LLMChain(llm=chat, prompt=sql_prompt_template)

    # Run the SQL generation chain
    sql_response = sql_generation_chain.run(
        user_input=user_input, schema_info=SCHEMA_INFO
    )

    # Extract the SQL query
    SQL_query = extract_sql_from_response(sql_response)
    say("Querying the database with:")
    say(SQL_query)
    # Execute the SQL query
    formatted_result, error = execute_bigquery_sql(SQL_query)
    say("Here's what I found:")
    say(f"```{formatted_result}```")

    if error:
        return f"Error executing SQL query: {error}"

    # 2. Final Response Chain
    final_prompt_template = PromptTemplate(
        input_variables=["user_query", "sql_result"],
        template="""
        # Context
        You are a helpful data analyst.

        # Table
        Use ONLY the data in the table below to answer the question. Do not use external knowledge.

        {sql_result}

        # User Question
        {user_query}

        # Instructions
        - Answer the question based only on the table above.
        - Be concise and specific.
        - If the answer is not clearly supported by the data, say so.
        """,
    )

    final_response_chain = LLMChain(llm=chat, prompt=final_prompt_template)

    # Run the final response chain
    return final_response_chain.run(user_query=user_input, sql_result=formatted_result)


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
