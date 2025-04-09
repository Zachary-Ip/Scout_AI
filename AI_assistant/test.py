import os

import openai
from dotenv import load_dotenv
import argparse

load_dotenv()  # Load from .env

client = openai.OpenAI(api_key=os.getenv("OPENAI_KEY"))


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
        "--prompt", type=str, required=True, help="Prompt to send to the OpenAI API"
    )
    args = parser.parse_args()

    output = ask_openai(args.prompt)
    print(f"\nResponse:\n{output}")


if __name__ == "__main__":
    main()
