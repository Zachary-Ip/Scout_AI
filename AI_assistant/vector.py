import chromadb
import pandas as pd
from sentence_transformers import SentenceTransformer

from dotenv import load_dotenv
import os

print("Starting vector process")
db_path = os.getenv("OPENAI_KEY")
print(db_path)
os.makedirs(db_path, exist_ok=True)

# Create a persistent client instead of the in-memory one
client = chromadb.PersistentClient(path=db_path)

print("Client created")
# Check if collection exists and create or get it
collection_name = "my_documents"
try:
    collection = client.get_collection(collection_name)
    print(f"Loaded existing collection with {collection.count()} documents")
except ValueError:
    # Collection doesn't exist yet, create it
    collection = client.create_collection(collection_name)
    print("Created new collection")


# For embedding generation without OpenAI (while you wait for credits)
model = SentenceTransformer("all-MiniLM-L6-v2")  # A good free alternative

# Load your dataset
df = pd.read_csv("data/laptops_dataset_final_600.csv")

# Process documents and generate embeddings
documents = []
embeddings = []
metadata = []
ids = []

# Assuming your CSV has a 'text' column and perhaps other metadata columns
for idx, row in df.iterrows():
    # Limit to the first 1000 for testing
    if idx >= 1000:
        break

    # Get the text content - adjust column name as needed
    text = row["review"]

    # Generate embedding using sentence-transformers
    embedding = model.encode(text)

    # Create a unique ID
    doc_id = f"doc_{idx}"

    # Store metadata (optional but useful)
    meta = {
        "source": "laptop_dataset",
        # Include other fields from your dataset that might be useful
        "title": row.get("title", ""),
        "product": row.get("product_name", ""),
        "rating": row.get("overall_rating", ""),
    }

    documents.append(text)
    embeddings.append(embedding)
    metadata.append(meta)
    ids.append(doc_id)

    # For large datasets, batch process
    if len(documents) >= 100:
        collection.add(
            documents=documents, embeddings=embeddings, metadatas=metadata, ids=ids
        )
        documents, embeddings, metadata, ids = [], [], [], []

# Add any remaining documents
if documents:
    collection.add(
        documents=documents, embeddings=embeddings, metadatas=metadata, ids=ids
    )


def update_vector_db(new_documents, collection):
    # Get existing IDs to avoid duplicates
    existing_ids = set()
    try:
        # This is a simplified approach - in practice you might
        # page through results for large collections
        all_results = collection.get(include=["documents", "metadatas", "embeddings"])
        existing_ids = set(all_results["ids"]) if "ids" in all_results else set()
    except Exception as e:
        print(f"Error getting existing IDs: {e}")

    # Process new documents
    documents = []
    embeddings = []
    metadatas = []
    ids = []

    for idx, doc in enumerate(new_documents):
        doc_id = f"doc_{doc['unique_identifier']}"  # Use a reliable unique ID

        # Skip if document already exists
        if doc_id in existing_ids:
            continue

        # Generate embedding
        embedding = model.encode(doc["text"])

        documents.append(doc["text"])
        embeddings.append(embedding)
        metadatas.append(doc["metadata"])
        ids.append(doc_id)

        # Batch process to avoid memory issues
        if len(documents) >= 100:
            collection.add(
                documents=documents, embeddings=embeddings, metadatas=metadatas, ids=ids
            )
            documents, embeddings, metadata, ids = [], [], [], []

    # Add any remaining documents
    if documents:
        collection.add(
            documents=documents, embeddings=embeddings, metadatas=metadatas, ids=ids
        )

    return len(new_documents) - len(existing_ids)  # Return number of new docs added


def query_vector_db(query_text, n_results=3):
    # Generate embedding for the query
    query_embedding = model.encode(query_text)

    # Search the collection
    results = collection.query(query_embeddings=[query_embedding], n_results=n_results)

    return results
