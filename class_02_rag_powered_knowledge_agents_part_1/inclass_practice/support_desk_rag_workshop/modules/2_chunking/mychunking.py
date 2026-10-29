"""AI CLASS LEARNING REFERENCE

Nine independent demonstrations of chunking and vector retrieval.
Run one example at a time by changing EXAMPLE_TO_RUN at the bottom.

Install packages:
    pip install langchain-openai langchain-experimental langchain-text-splitters langchain-core langchain-chroma numpy scikit-learn

For examples that use embeddings, set OPENAI_API_KEY in your environment.
Embeddings call the OpenAI API and may incur charges.
"""

import os

from dotenv import load_dotenv 
from openai import OpenAI 
import textwrap
from pathlib import Path

import numpy as np
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import (
    CharacterTextSplitter,
    HTMLHeaderTextSplitter,
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
root_env = Path(__file__).resolve().parents[5] / ".env"

# Load environment variables
load_dotenv(root_env)
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("OPENAI_API_KEY is missing")



EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
DB_DIRECTORY = Path(__file__).resolve().parent / "chroma_db"
DEMO_SENTENCES = [
    "Users cannot log in.",
    "Password reset is failing.",
    "Authentication service has errors.",
    "Database connections are timing out.",
    "Connection pool is exhausted.",
    "SQL queries are running slowly.",
]


# 1. FIXED-SIZE CHUNKING
# Concept: Cut text into character-sized pieces, even across word boundaries.
def fixed_size_chunking(
    text: str = "Apple Banana Mango Orange Grapes Pineapple",
    chunk_size: int = 12,
    overlap: int = 2,
    save_to_chroma: bool = True,
) -> list[str]:
    """Split by character count; optionally embed and store the resulting chunks."""
    # Step 1: Configure splitter. Empty separator allows mid-word splitting.
    splitter = CharacterTextSplitter(
        separator="", chunk_size=chunk_size, chunk_overlap=overlap
    )
    # Step 2: Produce and inspect chunks.
    chunks = splitter.split_text(text)
    print(f"\nFixed-size chunks: {len(chunks)}")
    for number, chunk in enumerate(chunks, start=1):
        print(f"Chunk {number}: {chunk!r} ({len(chunk)} characters)")
    # Step 3 (optional): Convert each chunk to its own Document and store.
    if save_to_chroma and chunks:
        documents = [Document(page_content=chunk) for chunk in chunks]
        embedding_model = OpenAIEmbeddings(model=EMBEDDING_MODEL)
        store = Chroma(
            collection_name="fixed_chunks",
            embedding_function=embedding_model,
            persist_directory=str(DB_DIRECTORY),
        )
        store.add_documents(documents, ids=[f"fixed_{i}" for i in range(len(documents))])
        print("Saved fixed-size chunks to ChromaDB.")
    return chunks


# 2. RECURSIVE CHARACTER CHUNKING
# Concept: Prefer paragraph, line, and word boundaries before splitting characters.
def recursive_chunking(
    text: str = "Apple Banana Mango Orange Grapes Pineapple",
    chunk_size: int = 12,
    overlap: int = 0,
    save_to_chroma: bool = True,
) -> list[str]:
    """Split using increasingly smaller natural boundaries."""
    # Step 1: Try separators in order until the text fits the target size.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    # Step 2: Split and display each actual chunk.
    chunks = splitter.split_text(text)
    print(f"\nRecursive chunks: {len(chunks)}")
    for number, chunk in enumerate(chunks, start=1):
        print(f"Chunk {number}: {chunk!r} ({len(chunk)} characters)")
    # Step 3 (optional): Store once, AFTER building all documents.
    if save_to_chroma and chunks:
        documents = [Document(page_content=chunk) for chunk in chunks]
        embedding_model = OpenAIEmbeddings(model=EMBEDDING_MODEL)
        store = Chroma(
            collection_name="recursive_chunks",
            embedding_function=embedding_model,
            persist_directory=str(DB_DIRECTORY),
        )
        store.add_documents(documents, ids=[f"recursive_{i}" for i in range(len(documents))])
        print("Saved recursive chunks to ChromaDB.")
    return chunks


# 3. MANUAL SEMANTIC CHUNKING
# Concept: Compare adjacent sentence embeddings with cosine similarity.
def manual_semantic_chunking(
    sentences: list[str] | None = None,
    threshold: float = 0.45,
) -> list[str]:
    """Start a new chunk whenever adjacent sentence similarity is below threshold."""
    if sentences is None:
        sentences = DEMO_SENTENCES.copy()
    if not sentences:
        return []
    if not -1 <= threshold <= 1:
        raise ValueError("Cosine similarity threshold must be between -1 and 1")
    # Step 1: Turn sentences into numerical embedding vectors.
    embedding_model = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    vectors = np.array(embedding_model.embed_documents(sentences))
    # Step 2: Compare consecutive sentences and build groups.
    chunks = []
    current_chunk = [sentences[0]]
    for index in range(len(sentences) - 1):
        similarity = cosine_similarity(
            vectors[index:index + 1], vectors[index + 1:index + 2]
        )[0][0]
        print(f"Sentence {index + 1} -> {index + 2}: similarity={similarity:.3f}")
        if similarity < threshold:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentences[index + 1]]
        else:
            current_chunk.append(sentences[index + 1])
    chunks.append(" ".join(current_chunk))
    # Step 3: Inspect and return the final groups.
    for number, chunk in enumerate(chunks, start=1):
        print(f"Chunk {number}: {chunk}")
    return chunks


# 4. LANGCHAIN SEMANTIC CHUNKING
# Concept: Use statistical distance breakpoints rather than a fixed similarity cutoff.
def semantic_chunking(
    text: str | None = None,
    threshold: float = 1.0,
    save_to_chroma: bool = True,
) -> list[str]:
    """Group sentences using LangChain's standard-deviation breakpoint strategy."""
    if text is None:
        text = " ".join(DEMO_SENTENCES)
    if not text.strip():
        return []
    # Step 1: Configure the embedding-based semantic splitter.
    embedding_model = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    splitter = SemanticChunker(
        embeddings=embedding_model,
        breakpoint_threshold_type="standard_deviation",
        breakpoint_threshold_amount=threshold,
        buffer_size=0,
    )
    # Step 2: Create and inspect chunks.
    chunks = splitter.split_text(text)
    print(f"\nLangChain semantic chunks: {len(chunks)}")
    for number, chunk in enumerate(chunks, start=1):
        print(f"Chunk {number}: {chunk}")
    # Step 3 (optional): Save chunks to a distinct Chroma collection.
    if save_to_chroma and chunks:
        documents = [Document(page_content=chunk) for chunk in chunks]
        store = Chroma(
            collection_name="semantic_chunks",
            embedding_function=embedding_model,
            persist_directory=str(DB_DIRECTORY),
        )
        store.add_documents(documents, ids=[f"semantic_{i}" for i in range(len(documents))])
        print("Saved semantic chunks to ChromaDB.")
    return chunks


# 5. MARKDOWN HEADER CHUNKING
# Concept: Split at Markdown headings and retain the heading hierarchy as metadata.
def markdown_chunking(
    text: str | None = None,
    save_to_chroma: bool = True,
) -> list[Document]:
    """Return LangChain Documents with content and Markdown header metadata."""
    if text is None:
        text = textwrap.dedent("""\
            # Database
            Database troubleshooting guide.

            ## Connection
            Check database connections.

            ### Timeout
            Increase connection timeout.

            ### Node Down
            One WebLogic node is down.
        """)
    # Step 1: Map heading levels to metadata names.
    headers = [("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3")]
    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers, strip_headers=False
    )
    # Step 2: The splitter returns Document objects (not plain strings).
    documents = splitter.split_text(text)
    for number, document in enumerate(documents, start=1):
        print(f"\nMarkdown chunk {number}")
        print("Content:", document.page_content)
        print("Metadata:", document.metadata)
    # Step 3 (optional): Embed and persist the documents.
    if save_to_chroma and documents:
        embedding_model = OpenAIEmbeddings(model=EMBEDDING_MODEL)
        store = Chroma(
            collection_name="markdown_chunks",
            embedding_function=embedding_model,
            persist_directory=str(DB_DIRECTORY),
        )
        store.add_documents(documents, ids=[f"markdown_{i}" for i in range(len(documents))])
        print("Saved Markdown chunks to ChromaDB.")
    return documents


# 6. HTML HEADER CHUNKING
# Concept: Split by h1/h2/h3 tags and preserve header metadata.
def html_chunking(
    text: str | None = None,
    save_to_chroma: bool = True,
) -> list[Document]:
    """Return Documents grouped by HTML headings and their metadata."""
    if text is None:
        text = """
        <html><body>
          <h1>Database Guide</h1><p>Database troubleshooting guide.</p>
          <h2>Connection Issues</h2><p>Check database connections.</p>
          <h3>Timeout Errors</h3><p>Increase connection timeout.</p>
          <h3>Node Down</h3><p>One WebLogic node is down.</p>
        </body></html>
        """
    # Step 1: Configure HTML heading hierarchy.
    headers = [("h1", "Header 1"), ("h2", "Header 2"), ("h3", "Header 3")]
    splitter = HTMLHeaderTextSplitter(headers_to_split_on=headers)
    # Step 2: Read content and metadata from each returned Document.
    documents = splitter.split_text(text)
    for number, document in enumerate(documents, start=1):
        print(f"\nHTML chunk {number}")
        print("Content:", document.page_content)
        print("Metadata:", document.metadata)
    # Step 3 (optional): Embed and persist the documents.
    if save_to_chroma and documents:
        embedding_model = OpenAIEmbeddings(model=EMBEDDING_MODEL)
        store = Chroma(
            collection_name="html_chunks",
            embedding_function=embedding_model,
            persist_directory=str(DB_DIRECTORY),
        )
        store.add_documents(documents, ids=[f"html_{i}" for i in range(len(documents))])
        print("Saved HTML chunks to ChromaDB.")
    return documents


# 7. CHROMA VECTOR STORE + SIMILARITY SEARCH
# Concept: Store embedded documents, then retrieve nearest matches.
def chroma_vector_store(
    documents: list[Document] | None = None,
    query: str = "Authentication problems after password reset",
    top_k: int = 3,
) -> Chroma:
    """Create a standalone Chroma collection, search it, and return the store."""
    # Step 1: Provide sample tickets unless custom documents were passed.
    if documents is None:
        documents = [
            Document(page_content="Users cannot log in after password reset.",
                     metadata={"ticket_id": "TICK-001", "category": "Authentication"}),
            Document(page_content="Database connection timeout in production.",
                     metadata={"ticket_id": "TICK-002", "category": "Database"}),
            Document(page_content="Password reset link expired.",
                     metadata={"ticket_id": "TICK-003", "category": "Authentication"}),
        ]
    if not documents:
        raise ValueError("documents cannot be empty")
    # Step 2: Create a vector collection and embed the documents.
    embedding_model = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    store = Chroma(
        collection_name="whole_documents_demo",
        embedding_function=embedding_model,
        persist_directory=str(DB_DIRECTORY),
    )
    ids = [f"whole_ticket_{i}" for i in range(len(documents))]
    store.add_documents(documents, ids=ids)
    # Step 3: Find closest matches. Chroma returns a distance; lower is closer.
    results = store.similarity_search_with_score(query=query, k=top_k)
    for number, (document, distance) in enumerate(results, start=1):
        print(f"\nResult {number}: {document.page_content}")
        print("Metadata:", document.metadata)
        print(f"Distance: {distance:.4f}")
    return store


# 8. MAXIMAL MARGINAL RELEVANCE (MMR)
# Concept: Retrieve results that balance relevance and diversity.
def mmr_search(
    documents: list[Document] | None = None,
    query: str = "Authentication problems after password reset",
    top_k: int = 2,
    fetch_k: int = 3,
    lambda_mult: float = 0.5,
) -> list[Document]:
    """Demonstrate MMR using its OWN documents and Chroma collection."""
    # k: number of final documents.
    # fetch_k: number of candidates considered (should be >= k).
    # lambda_mult: 1.0 favors relevance; 0.0 favors diversity.
    if documents is None:
        documents = [
            Document(page_content="Users cannot log in after password reset.",
                     metadata={"ticket_id": "TICK-001", "category": "Authentication"}),
            Document(page_content="Database connection timeout in production.",
                     metadata={"ticket_id": "TICK-002", "category": "Database"}),
            Document(page_content="Password reset link expired.",
                     metadata={"ticket_id": "TICK-003", "category": "Authentication"}),
        ]
    if not documents:
        raise ValueError("documents cannot be empty")
    if fetch_k < top_k:
        raise ValueError("fetch_k must be >= top_k")
    if not 0 <= lambda_mult <= 1:
        raise ValueError("lambda_mult must be between 0 and 1")
    # Step 1: Set up this concept's independent vector collection.
    embedding_model = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    store = Chroma(
        collection_name="support_tickets_mmr_demo",
        embedding_function=embedding_model,
        persist_directory=str(DB_DIRECTORY),
    )
    store.add_documents(documents, ids=[f"mmr_ticket_{i}" for i in range(len(documents))])
    # Step 2: Fetch candidate matches and select a diverse subset.
    results = store.max_marginal_relevance_search(
        query=query, k=top_k, fetch_k=fetch_k, lambda_mult=lambda_mult
    )
    for number, document in enumerate(results, start=1):
        print(f"\nMMR result {number}: {document.page_content}")
        print("Metadata:", document.metadata)
    return results


# 9. CHROMA METADATA FILTERING
# Concept: Restrict the search to documents with matching structured metadata.
def metadata_filtering(
    documents: list[Document] | None = None,
    query: str = "Login issues after password reset",
    category: str = "Authentication",
    top_k: int = 3,
) -> list[Document]:
    """Compare similarity search with and without a metadata filter."""
    if documents is None:
        documents = [
            Document(page_content="Users cannot log in after password reset.",
                     metadata={"ticket_id": "TICK-001", "category": "Authentication", "priority": "High"}),
            Document(page_content="Database connection timeout in production.",
                     metadata={"ticket_id": "TICK-002", "category": "Database", "priority": "High"}),
            Document(page_content="Password reset link expired.",
                     metadata={"ticket_id": "TICK-003", "category": "Authentication", "priority": "Medium"}),
            Document(page_content="User cannot log in because MFA token expired.",
                     metadata={"ticket_id": "TICK-004", "category": "Authentication", "priority": "High"}),
        ]
    if not documents:
        raise ValueError("documents cannot be empty")
    # Step 1: Store this concept's own tickets and their metadata.
    embedding_model = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    store = Chroma(
        collection_name="support_tickets_metadata_demo",
        embedding_function=embedding_model,
        persist_directory=str(DB_DIRECTORY),
    )
    store.add_documents(documents, ids=[f"metadata_ticket_{i}" for i in range(len(documents))])
    # Step 2: Compare unfiltered similarity search with filtered search.
    all_results = store.similarity_search(query=query, k=top_k)
    print("\nWITHOUT metadata filter:")
    for document in all_results:
        print(document.metadata.get("category"), "-", document.page_content)
    filtered_results = store.similarity_search(
        query=query, k=top_k, filter={"category": category}
    )
    print(f"\nWITH category={category!r} filter:")
    for document in filtered_results:
        print(document.metadata.get("category"), "-", document.page_content)
    return filtered_results


# RUN ONE OR ALL EXAMPLES
# Use 1 through 9 for one lesson, "ALL" for every lesson, or None for none.
EXAMPLE_TO_RUN = "ALL"


def main() -> None:
    """Optional launcher; the nine learning functions never call one another."""
    examples = {
        1: fixed_size_chunking,
        2: recursive_chunking,
        3: manual_semantic_chunking,
        4: semantic_chunking,
        5: markdown_chunking,
        6: html_chunking,
        7: chroma_vector_store,
        8: mmr_search,
        9: metadata_filtering,
    }
    if isinstance(EXAMPLE_TO_RUN, str) and EXAMPLE_TO_RUN.upper() == "ALL":
        for number, example in examples.items():
            print(f"\n========== EXAMPLE {number}: {example.__name__} ========== ")
            example()
    elif EXAMPLE_TO_RUN is not None:
        if EXAMPLE_TO_RUN not in examples:
            raise ValueError('EXAMPLE_TO_RUN must be 1 through 9, "ALL", or None')
        examples[EXAMPLE_TO_RUN]()


if __name__ == "__main__":
    main()
