import argparse
import os

import chromadb
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

db_path = os.getenv("DB_PATH")

collection_name = os.getenv("COLLECTION_NAME")

model = SentenceTransformer("all-MiniLM-L6-v2")


print("Starting query process")
try:
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_collection(collection_name)
    print(f"Loaded existing collection with {collection.count()} documents")
except chromadb.errors.NotFoundError:
    print("There doesn't seem to be a DB to query, exiting..")


def main():

    parser = argparse.ArgumentParser(description="Send a prompt to OpenAI's API.")
    parser.add_argument(
        "--prompt", type=str, required=True, help="Prompt to send to the OpenAI API"
    )
    parser.add_argument("--nres", type=int, required=False, default=3)
    args = parser.parse_args()

    results = query_vector_db(args.prompt, args.nres)

    print(f"\nResponse:\n{results["documents"]}")


def query_vector_db(query_text, n_results=3):
    # Generate embedding for the query
    query_embedding = model.encode(query_text)

    # Search the collection
    results = collection.query(query_embeddings=[query_embedding], n_results=n_results)

    return results


if __name__ == "__main__":
    main()
