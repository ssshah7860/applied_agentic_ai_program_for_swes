"""CLASS 02 — EMBEDDINGS AND SEMANTIC SEARCH | LEARNING REFERENCE

PURPOSE
-------
Read historical support tickets from JSON, turn their text into numeric vectors
(embeddings), compare a user's question with the ticket vectors using cosine
similarity, and display the most similar tickets.

WORKFLOW
--------
JSON tickets -> searchable text -> document embeddings
User question -> query embedding -> cosine similarity -> top K tickets

KEY IDEAS
---------
* Embedding: a list of numbers representing the meaning of text.
* NumPy array: a convenient 2-D numeric container for matrix calculations.
* Cosine similarity: measures how aligned two vectors are (higher = closer).
* top_k: how many matching tickets we want to return.

EXAMPLE DATA SHAPES (text-embedding-3-small, default 1536 dimensions)
--------------------------------------------------------------------
3 tickets -> Python list of 3 embedding lists -> NumPy shape (3, 1536)
1 question -> Python list of 1 embedding list -> NumPy shape (1, 1536)
cosine_similarity(question, tickets) -> shape (1, 3)
[0] extracts the first/only row -> shape (3,)

INSTALL
-------
python -m pip install openai python-dotenv numpy scikit-learn

CONFIGURATION
-------------
OPENAI_API_KEY must be available in your environment or an accessible .env file.
Optional: OPENAI_EMBEDDING_MODEL, SUPPORT_TICKETS_JSON.
OpenAI embedding calls use the API and may incur charges.

Run: python myembeddings_reviewed.py
"""

# ============================================================
# SECTION 1 — IMPORTS
# ============================================================
import json                     # Read JSON file into Python objects
import os                       # Read environment variables
from pathlib import Path        # Build cross-platform paths (Windows-friendly)

import numpy as np              # Store embeddings in 2-D numeric arrays
from dotenv import load_dotenv  # Load variables from .env files
from openai import OpenAI       # Call OpenAI's embedding endpoint
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# 1. LOAD ENVIRONMENT CONFIGURATION
# ============================================================
def load_env(env_path: Path | None = None) -> None:
    """Load OpenAI settings from .env, if one is found.

    WHY?
        API keys should live outside Python source code, not inside a script.

    INPUT:
        env_path: explicit location of a .env file, or None to search upward.

    OUTPUT:
        None. Values become available through os.getenv(...).

    NOTE:
        If the key is already configured in Windows/environment settings,
        no .env file is necessary.
    """
    if env_path is not None:
        if not env_path.is_file():
            raise FileNotFoundError(f"Missing .env file: {env_path}")
        load_dotenv(env_path)
        return

    # Search for .env beside this file, then in its parent folders.
    script_dir = Path(__file__).resolve().parent
    for folder in (script_dir, *script_dir.parents):
        candidate = folder / ".env"
        if candidate.is_file():
            load_dotenv(candidate)
            print(f"Loaded environment configuration: {candidate}")
            return

    # It is okay to have no .env if OPENAI_API_KEY is in the OS environment.


# ============================================================
# 2. CREATE AN OPENAI CLIENT
# ============================================================
def get_client() -> OpenAI:
    """Create a client for making OpenAI API requests.

    WHY?
        The client handles authenticated API calls, including embeddings.

    INPUT: None (reads OPENAI_API_KEY from the environment).
    OUTPUT: OpenAI client object; raises a clear error if the key is missing.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is missing. Add it to your .env or OS environment.")
    return OpenAI(api_key=api_key)


# ============================================================
# 3. LOAD HISTORICAL SUPPORT TICKETS
# ============================================================
def read_knowledge_records(file_path: Path) -> list[dict]:
    """Read a JSON array of support ticket dictionaries.

    WHY?
        We need real text records before creating any embeddings.

    EXAMPLE JSON: [{"ticket_id": "T1", "title": "Login failed", ...}]
    OUTPUT TYPE: list[dict] (one dictionary per support ticket).
    """
    with file_path.open("r", encoding="utf-8") as file:
        #converts the JSON into Python objects eg list of dict 
        records = json.load(file)

    if not isinstance(records, list):
        raise ValueError("Expected a JSON list of ticket records")
    if not all(isinstance(record, dict) for record in records):
        raise ValueError("Every ticket in the JSON list must be a dictionary")
    if not records:
        raise ValueError("The support ticket JSON file contains no records")
    return records


# ============================================================
# 4. TURN EACH TICKET DICTIONARY INTO SEARCHABLE TEXT
# ============================================================
def convert_records_to_text(records: list[dict]) -> list[str]:
    """Create one human-readable text string per support ticket.

    WHY?
        Embedding models accept text, not arbitrary Python dictionaries.
        Combining title + description + resolution supplies useful context.

    INPUT: list of ticket dictionaries.
    OUTPUT: list of strings, with the SAME order as input records.

    IMPORTANT:
        We preserve list order so search result index 0 still refers to
        the first original ticket (records[0]).
    """
    documents = []
    for record in records:
        text = (
            f"Title: {record.get('title', '')}\n"
            f"Description: {record.get('description', '')}\n"
            f"Resolution: {record.get('resolution', '')}"
        )
        documents.append(text)
    return documents


# ============================================================
# 5. GENERATE EMBEDDINGS AND CONVERT TO A NUMPY ARRAY
# ============================================================
def generate_embeddings(client: OpenAI, texts: list[str], model: str) -> np.ndarray:
    """Send text to OpenAI and return a 2-D NumPy array of embedding vectors.

    WHY generate embeddings?
        Cosine similarity compares numbers, not raw English sentences.
        The embedding model converts each text into a numeric vector.

    WHY use np.array(vectors)?
        response.data contains one embedding (a Python list of floats)
        for each input text.  The list comprehension below collects them:
            vectors = [[0.1, 0.2, ...], [0.3, 0.4, ...], ...]
        np.array(vectors) creates a 2-D numeric matrix, which makes it easy to:
          * inspect .shape, e.g. (20, 1536) for 20 tickets;
          * select rows with NumPy slicing;
          * use efficient numeric operations with NumPy/scikit-learn.

        Important nuance: scikit-learn cosine_similarity ALSO accepts ordinary
        nested Python lists. Conversion is not strictly mandatory; it makes
        numeric data easier to work with and its shape explicit.

    INPUT: client, list[str] texts, embedding model name.
    OUTPUT: np.ndarray, shape (number_of_texts, embedding_dimensions).
    """
    if not texts:
        raise ValueError("Cannot generate embeddings for an empty text list")

    # API returns an object with .data; each item has its own .embedding list.
    response = client.embeddings.create(input=texts, model=model)

    # Keep API response order aligned with the input texts.
    ordered_items = sorted(response.data, key=lambda item: item.index)
    vectors = [item.embedding for item in ordered_items]

    # Example with tiny *illustrative* vectors:
    # vectors = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]  # plain Python list
    # np.array(vectors).shape == (2, 3)             # 2 rows, 3 columns
    embedding_matrix = np.array(vectors, dtype=float)
    print("Embedding matrix shape:", embedding_matrix.shape)
    return embedding_matrix


# ============================================================
# 6. FIND THE TOP-K MOST SIMILAR DOCUMENTS
# ============================================================
def search_similar_documents(
    query_embedding: np.ndarray,
    document_embeddings: np.ndarray,
    top_k: int = 3,
) -> list[tuple[int, float]]:
    """Return (ticket_index, similarity_score) pairs, best score first.

    WHY?
        Retrieval ranks existing tickets by meaning compared with the question.

    INPUT EXAMPLE:
        query_embedding.shape = (1, 1536)  -> one question
        document_embeddings.shape = (20, 1536) -> twenty tickets
    OUTPUT EXAMPLE:
        [(4, 0.91), (2, 0.85)] -> indexes into ORIGINAL records, with scores

    NUMPY/SKLEARN EXPLANATION:
        cosine_similarity(query_embedding, document_embeddings)
            returns shape (1, 20): one row of 20 scores.
        [0] -> extract that row: shape (20,).
        np.argsort(scores) -> indexes sorted from LOWEST to HIGHEST score.
        [::-1] -> reverse to HIGHEST first.
        [:top_k] -> take only the best K indexes.
    """
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    if query_embedding.ndim != 2 or query_embedding.shape[0] != 1:
        raise ValueError("Expected query_embedding shape (1, dimensions)")
    if document_embeddings.ndim != 2:
        raise ValueError("Expected document_embeddings to be a 2-D array")

    similarity_matrix = cosine_similarity(query_embedding, document_embeddings)
    similarities = similarity_matrix[0]  # shape (number_of_documents,)
    print("Similarity scores:", np.round(similarities, 4))

    sorted_indices = np.argsort(similarities)[::-1]  # highest score first
    top_indices = sorted_indices[:top_k]

    # int/float convert NumPy scalar types into ordinary Python types.
    results = []
    for idx in top_indices:
        results.append((int(idx), float(similarities[idx])))
    return results


# ============================================================
# 7. DISPLAY TOP-K MATCHES WITH ORIGINAL TICKET FIELDS
# ============================================================
def display_results(results: list[tuple[int, float]], records: list[dict]) -> None:
    """Print retrieved tickets with their similarity score and details.

    WHY?
        A similarity score alone isn't helpful for troubleshooting.
        The indexes map back to the original JSON records; we print their fields.

    Note: enumerate(..., start=1) gives human-friendly rank 1, 2, 3 rather
    than Python's 0, 1, 2 indexing.
    """
    for rank, (idx, score) in enumerate(results, start=1):
        ticket = records[idx]
        print(f"\nRank: {rank}")
        print(f"Cosine similarity: {score:.4f}")  # four digits after decimal
        print(f"Ticket ID: {ticket.get('ticket_id', 'Unknown')}")
        print(f"Title: {ticket.get('title', '')}")
        print(f"Description: {ticket.get('description', '')}")


# ============================================================
# 8. MAIN WORKFLOW — PUT THE STEPS TOGETHER
# ============================================================
def main() -> None:
    """Run ingestion -> embedding -> similarity search -> display.

    Keep the individual functions small, each with one responsibility.
    This main() function is ONLY the coordinator of those steps.
    """
    load_env()
    client = get_client()
    model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

    # Optional override to point to your Class 02 synthetic_tickets.json.
    # If the variable is not set, try a few common locations near this file.
    configured_path = os.getenv("SUPPORT_TICKETS_JSON")
    script_dir = Path(__file__).resolve().parent
    candidate_paths = ([Path(configured_path).expanduser()] if configured_path else [
        script_dir / "synthetic_tickets.json",
        script_dir / "data" / "synthetic_tickets.json",
        script_dir.parent / "data" / "synthetic_tickets.json",
        script_dir.parent.parent / "data" / "synthetic_tickets.json",
    ])
    data_path = next((p for p in candidate_paths if p.is_file()), None)
    if data_path is None:
        raise FileNotFoundError(
            "Cannot locate synthetic_tickets.json. Set SUPPORT_TICKETS_JSON "
            "to the full path of your dataset in your .env file."
        )

    records = read_knowledge_records(data_path)
    documents = convert_records_to_text(records)
    print(f"Loaded {len(records)} support tickets from: {data_path}")

    # ONE embedding per document, arranged as rows in a NumPy matrix.
    document_embeddings = generate_embeddings(client, documents, model)

    # [query] is a LIST containing one string: the API expects multiple inputs.
    # This gives a 2-D matrix of shape (1, embedding_dimensions).
    query = "Users can't login after changing password"
    query_embedding = generate_embeddings(client, [query], model)

    matches = search_similar_documents(
        query_embedding, document_embeddings, top_k=2
    )
    display_results(matches, records)


if __name__ == "__main__":
    # This block runs when you execute the file directly, not when imported.
    main()
