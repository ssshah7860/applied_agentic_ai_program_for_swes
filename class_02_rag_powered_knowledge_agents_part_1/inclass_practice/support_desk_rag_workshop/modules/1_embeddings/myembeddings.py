import os

from dotenv import load_dotenv 
from openai import OpenAI       # OpenAI API client for generating embeddings
import json
import numpy as np              # For numerical operations on embedding vectors
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity  # Measure similarity between vectors
import ast

# 1. Load environment configuration
def load_env(env_path: Path):
    if not env_path.is_file():
        raise FileNotFoundError(f"Missing .env file: {env_path}")

    load_dotenv(env_path)


# 2. Initialize OpenAI client
def get_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise ValueError("OPENAI_API_KEY is missing")

    return OpenAI(api_key=api_key)

# 3. Read JSON knowledge records
def read_knowledge_records(file_path: Path) -> list[dict]:
    with open(file_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    if not isinstance(records, list):
        raise ValueError("Expected a JSON list of records")

    return records

# 4. Convert records into meaningful text
def convert_records_to_text(records: list[dict]) -> list[str]:
    documents = []

    for record in records:
        text = (
            f"Title: {record.get('title', '')}\n"
            f"Description: {record.get('description', '')}\n"
            f"Resolution: {record.get('resolution', '')}"
        )
        documents.append(text)

    return documents


# 5. Generate embeddings for any list of texts
def generate_embeddings(
    client: OpenAI,
    texts: list[str],
    model: str
) -> np.ndarray:

    if not texts:
        raise ValueError("Cannot generate embeddings for empty texts")

    response = client.embeddings.create(
        input=texts,
        model=model
    )

    vectors = [item.embedding for item in response.data]

    return np.array(vectors)

# 6. Search most similar documents
def search_similar_documents(
    query_embedding: np.ndarray,
    document_embeddings: np.ndarray,
    top_k: int = 3
) -> list[tuple[int, float]]:

    if top_k < 1:
        raise ValueError("top_k must be at least 1")

    similarities = cosine_similarity(
        query_embedding,
        document_embeddings
    )[0]

    top_indices = np.argsort(similarities)[::-1][:top_k]

    results = []

    for idx in top_indices:
        results.append((int(idx), float(similarities[idx])))

    return results

# 7. Display results
def display_results(results, records):
    for rank, (idx, score) in enumerate(results, start=1):
        ticket = records[idx]

        print(f"\nRank: {rank}")
        print(f"Score: {score:.4f}")
        print(f"Ticket ID: {ticket.get('ticket_id')}")
        print(f"Title: {ticket.get('title')}")
        print(f"Description: {ticket.get('description')}")

# Main workflow
def main():
    # Adjust these paths to match your project structure
    #project_root = Path(__file__).resolve().parent
    project_root = Path(__file__).resolve()
    env_path = project_root.parents[5]  / ".env"
    data_path = project_root.parent / "../../data/synthetic_tickets.json"

    load_env(env_path)

    client = get_client()
    model = os.getenv(
        "OPENAI_EMBEDDING_MODEL",
        "text-embedding-3-small"
    )

    records = read_knowledge_records(data_path)
    documents = convert_records_to_text(records)

    # Embed documents
    document_embeddings = generate_embeddings(
        client, documents, model
    )

    # Embed user query
    query = "Users can't login after changing password"

    query_embedding = generate_embeddings(
        client, [query], model
    )

    # Search top 2 tickets
    results = search_similar_documents(
        query_embedding,
        document_embeddings,
        top_k=2
    )

    display_results(results, records)


if __name__ == "__main__":
    main()
