import argparse
import os

import openai
from dotenv import load_dotenv

load_dotenv()  # Load from .env


client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def ask_openai(prompt: str) -> str:
    """Send a prompt to OpenAI and return the response."""
    response = client.chat.completions.create(
        model="gpt-3.5-turbo", messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()


def main():
    print("HELLO")
    print(os.getenv("OPENAI_KEY"))
    parser = argparse.ArgumentParser(description="Send a prompt to OpenAI's API.")
    parser.add_argument(
        "--prompt",
        type=str,
        required=False,
        help="Prompt to send to the OpenAI API",
        default="""
    Given the following information:
    Row(('bolivia vs uruguay', 1.0, 55020), {'term': 0, 'avg_rank': 1, 'num_entries': 2})
    Row(('groundhog day 2025', 1.0, 55020), {'term': 0, 'avg_rank': 1, 'num_entries': 2})
    Row(('xavier worthy', 1.0, 54810), {'term': 0, 'avg_rank': 1, 'num_entries': 2})
    Row(('ted lasso season 4', 1.0, 54810), {'term': 0, 'avg_rank': 1, 'num_entries': 2})
    Row(('angie stone', 1.0, 54810), {'term': 0, 'avg_rank': 1, 'num_entries': 2})
    Row(('michelle trachtenberg', 1.0, 55020), {'term': 0, 'avg_rank': 1, 'num_entries': 2})
    Row(('earthquake', 1.0, 55020), {'term': 0, 'avg_rank': 1, 'num_entries': 2})
    Row(('f-47', 1.0, 54810), {'term': 0, 'avg_rank': 1, 'num_entries': 2})
    Row(('jeffrey goldberg', 1.0, 55020), {'term': 0, 'avg_rank': 1, 'num_entries': 2})
    Row(('zelensky', 1.0, 54810), {'term': 0, 'avg_rank': 1, 'num_entries': 2})

    answer the following query:

    What are the trends of the top search terms from this week?
""",
    )
    args = parser.parse_args()

    output = ask_openai(args.prompt)
    print(f"\nResponse:\n{output}")


if __name__ == "__main__":
    main()
