import os

from dotenv import find_dotenv, load_dotenv
from google.cloud import bigquery
from langchain.chains import LLMChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_community.chat_models import ChatOpenAI
from tabulate import tabulate

load_dotenv(find_dotenv())
OPENAI_API_KEY = os.getenv("OPENAI_KEY")
SCHEMA_INFO = """
Table: top_terms
Columns:
- refresh_date (DATE): The date the data was refreshed
- dma_name (STRING): The name of the designated market area
- dma_id (INTEGER): The numeric ID of the DMA
- term (STRING): The search term
- week (DATE): The week of the trend data
- score (INTEGER): The popularity score of the term
- rank (INTEGER): The rank of the term for that week and DMA
"""


def draft_email(user_input):
    chat = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=1)

    template = """
    
    You are a helpful assistant that drafts an email reply based on an a new email.
    
    You goal is help the user quickly create a perfect email reply by.
    
    Keep your reply short and to the point and mimic the style of the email so you reply in a similar manner to match the tone.
    
    Make sure to sign of with {signature}.
    
    """

    signature = "Kind regards, \n\nDave"
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)

    human_template = "Here's the email to reply to and consider any other comments from the user for reply as well: {user_input}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chain = LLMChain(llm=chat, prompt=chat_prompt)
    response = chain.run(user_input=user_input, signature=signature)

    return response


def RAG_response(user_input):
    chat = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=1)
    SQL_prompt = create_sql_generation_prompt(user_input, SCHEMA_INFO)

    chain = LLMChain(llm=chat, prompt=SQL_prompt)
    response = chain.run(user_input=SQL_prompt)
    SQL_query = extract_sql_from_response(response)

    SQL_result = execute_bigquery_sql(SQL_query)

    final_prompt = create_final_prompt(user_input, SQL_result)

    return chain.run(user_input=final_prompt)


def create_sql_generation_prompt(user_query, schema_info):
    return f"""
    # Data source
    FROM `bigquery-public-data.google_trends.top_terms`

    # Database Schema
    {schema_info}
    
    # User Query
    "{user_query}"
    
    # Instructions
     
    Write a syntactically valid BigQuery Standard SQL query that answers the user’s question using only the schema provided. 
    Do not include any data modification statements (e.g., INSERT, UPDATE, DELETE).
    Return only the SQL query, wrapped in ```sql``` tags — no additional text, explanation, or formatting.

    """


def create_final_prompt(user_query, SQL_result):
    return f"""
    # Context
    You are a helpful data analyst.

    # Table
    Use ONLY the data in the table below to answer the question. Do not make assumptions or use external knowledge.

    {SQL_result}

    # User Question
    {user_query}

    # Instructions
    - Answer the question based only on the table above.
    - Be concise and specific.
    - If the answer is not clearly supported by the data, say so.
    """


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


os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.getenv(
    "GOOGLE_APPLICATION_CREDENTIALS"
)


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
