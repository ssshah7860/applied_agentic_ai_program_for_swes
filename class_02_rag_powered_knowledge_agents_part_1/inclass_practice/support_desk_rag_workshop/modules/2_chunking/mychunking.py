"""CLASS 02 RAG PART 1 | CHUNKING LEARNING REFERENCE

PURPOSE
-------
Review one RAG concept at a time. Every numbered function can be called
independently; none of the nine calls another example function.

1. Fixed-size character chunking
2. Recursive character chunking
3. Manual semantic chunking (cosine threshold)
4. LangChain semantic chunking (statistical breakpoints)
5. Markdown heading chunking
6. HTML heading chunking
7. Chroma vector store and similarity search
8. Maximum Marginal Relevance (MMR)
9. Chroma metadata filtering

Run EXAMPLE_TO_RUN = 1 through 9, or "ALL", or None (bottom of file).

Install:
  pip install python-dotenv numpy scikit-learn langchain-core \
      langchain-openai langchain-chroma langchain-experimental \
      langchain-text-splitters lxml

Examples 1, 2, 5, 6 can run offline with save_to_chroma=False.
Examples 3, 4, 7, 8, 9, and any save_to_chroma=True require an OpenAI API key.
OpenAI embedding requests may incur charges.
"""

import os
import textwrap
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
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


# CONFIGURATION: These constants are shared settings, not other example functions.
# The .env file is located at the project root in your existing folder layout.
ENV_PATH = Path(__file__).resolve().parents[5] / ".env" if len(Path(__file__).resolve().parents) > 5 else Path(__file__).resolve().parent / ".env"
load_dotenv(ENV_PATH)
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


# ============================================================
# 1. FIXED-SIZE CHUNKING
# WHAT: Split text into chunks based on number of characters.
# WHY: Smaller pieces can later be embedded and searched in RAG.
# NOTE: Empty separator means words can be cut in the middle.
# ============================================================
def fixed_size_chunking(
    text: str = "Apple Banana Mango Orange Grapes Pineapple",
    chunk_size: int = 12,
    overlap: int = 2,
    save_to_chroma: bool = True,
) -> list[str]:
    """Return character-sized chunks; optionally save them to Chroma."""
    # Validate inputs: overlap cannot be equal to or larger than chunk_size.
    if chunk_size <= 0 or not 0 <= overlap < chunk_size:
        raise ValueError("Require chunk_size > 0 and 0 <= overlap < chunk_size")

    # Step 1: Create a splitter object with fixed character limits.
    splitter = CharacterTextSplitter(
        separator="", chunk_size=chunk_size, chunk_overlap=overlap
    )

    # Step 2: split_text returns list[str], NOT embedding vectors.
    chunks = splitter.split_text(text)
    print(f"\nFixed-size chunks ({len(chunks)}):")
    for number, chunk in enumerate(chunks, start=1):
        print(f"Chunk {number}: {chunk!r} ({len(chunk)} characters)")

    # Step 3: OPTIONAL separate activity: embed and persist chunks in Chroma.
    # The chunking concept does not need Chroma or an API call.
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


# ============================================================
# 2. RECURSIVE CHARACTER CHUNKING
# WHAT: Try paragraph, newline, space, then character boundaries.
# WHY: Keep related words/sentences together whenever possible.
# ============================================================
def recursive_chunking(
    text: str = "Apple Banana Mango Orange Grapes Pineapple",
    chunk_size: int = 12,
    overlap: int = 0,
    save_to_chroma: bool = True,
) -> list[str]:
    """Return chunks split on natural boundaries when possible."""
    if chunk_size <= 0 or not 0 <= overlap < chunk_size:
        raise ValueError("Require chunk_size > 0 and 0 <= overlap < chunk_size")

    # Step 1: Separators are tried from larger boundaries to smaller ones.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    # Step 2: Obtain the actual pieces of text.
    chunks = splitter.split_text(text)
    print(f"\nRecursive chunks ({len(chunks)}):")
    for number, chunk in enumerate(chunks, start=1):
        print(f"Chunk {number}: {chunk!r} ({len(chunk)} characters)")

    # Step 3: OPTIONAL: save this function's chunks without calling another demo.
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


# ============================================================
# 3. MANUAL SEMANTIC CHUNKING
# WHAT: Compare each sentence embedding with the next one.
# WHY: A low similarity suggests a change of topic -> split there.
# DIFFERENCE FROM 4: You manually choose an absolute cosine threshold.
# ============================================================

import os
import numpy as np

from pathlib import Path
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity

from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_chroma import Chroma


# ============================================================
# 3. MANUAL SEMANTIC CHUNKING
# ============================================================
#
# WHAT?
#   Group sentences into chunks based on their meaning.
#
# WHY?
#   Sentences discussing similar topics should stay together.
#   When the meaning changes, we create a new chunk.
#
# HOW?
#   1. Generate embeddings for all sentences.
#   2. Compare consecutive sentences using cosine similarity.
#   3. Create a new chunk when similarity is below threshold.
#   4. Convert final chunks into LangChain Documents.
#   5. Save final chunks and their embeddings into ChromaDB.
#
# EXAMPLE:
#   Sentence 1: Users cannot log in.
#   Sentence 2: Password reset is failing.
#   Sentence 3: Database connections are timing out.
#
#   Sentences 1 and 2 discuss authentication.
#   Sentence 3 discusses database problems.
#
#   Possible result:
#       Chunk 1: Users cannot log in. Password reset is failing.
#       Chunk 2: Database connections are timing out.
#
# NOTE:
#   Actual grouping depends on embedding similarity scores.
#
# ============================================================

def manual_semantic_chunking(
    sentences: list[str] | None = None,
    threshold: float = 0.45,
    save_to_chroma: bool = True,
) -> list[str]:
    """Group adjacent sentences; split when cosine similarity is low."""
    if sentences is None:
        sentences = DEMO_SENTENCES.copy()
    if not sentences:
        return []
    if not -1 <= threshold <= 1:
        raise ValueError("threshold must be between -1 and 1")

    # Step 1: Embedding model converts sentences into numerical vectors.
    embedding_model = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    embedding_lists = embedding_model.embed_documents(sentences)
    vectors = np.array(embedding_lists)  # 2D NumPy array: (sentences, dimensions)
    print("Embedding array shape:", vectors.shape)

    # Step 2: Keep the first sentence in the current chunk.
    chunks = []
    current_chunk = [sentences[0]]

    # Step 3: Compare pairs (sentence 1 with 2, then 2 with 3, etc.).
    for index in range(len(sentences) - 1):
        # Slices [index:index+1] preserve NumPy's 2D shape (1, dimensions).
        similarity_matrix = cosine_similarity(
            vectors[index:index + 1], vectors[index + 1:index + 2]
        )
        similarity = float(similarity_matrix[0][0])  # Extract one score.
        print(f"Sentence {index + 1} -> {index + 2}: similarity={similarity:.3f}")

        if similarity < threshold:
            # Topic probably changed: close current chunk and start a new one.
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentences[index + 1]]
        else:
            # Topic is similar: add the next sentence to the same chunk.
            current_chunk.append(sentences[index + 1])

    # Step 4: Preserve the final chunk (the loop does not append it).
    chunks.append(" ".join(current_chunk))
    for number, chunk in enumerate(chunks, start=1):
        print(f"Chunk {number}: {chunk}")
    if save_to_chroma:

        documents = [
            Document(
                page_content=chunk,
                metadata={
                    "chunk_number": index + 1,
                    "chunking_method": "manual_semantic"
                }
            )
            for index, chunk in enumerate(chunks)
        ]

        db_path = Path(__file__).resolve().parent / "chroma_db"

        store = Chroma(
            collection_name="manual_semantic_chunks",
            embedding_function=embedding_model,
            persist_directory=str(db_path)
        )

        store.add_documents(
            documents=documents,
            ids=[
                f"manual_semantic_{index}"
                for index in range(len(documents))
            ]
        )

        print(
            f"\nSaved {len(documents)} chunks into ChromaDB."
        )

        # ----------------------------------------------------
        # STEP 11: Verify stored documents
        # ----------------------------------------------------
        # WHY?
        #   Confirm that the collection contains the chunks.
        #
        # collection.get() retrieves stored records.
        #
        # NOTE:
        #   Existing records from earlier runs may also be
        #   present because this is a persistent collection.

        stored_records = store.get(
            include=["documents", "metadatas"]
        )

        print("\nDocuments stored in ChromaDB:")

        for doc_id, text, metadata in zip(
            stored_records["ids"],
            stored_records["documents"],
            stored_records["metadatas"]
        ):
            print("ID:", doc_id)
            print("Text:", text)
            print("Metadata:", metadata)
            print("---")

    # --------------------------------------------------------
    # STEP 12: Return the final chunks
    # --------------------------------------------------------
    # Return type: list[str].
    #
    # The function returns chunks regardless of whether
    # save_to_chroma is True or False.

    return chunks






# ============================================================
# 4. LANGCHAIN SEMANTIC CHUNKING
# WHAT: Split using unusually large embedding distance changes.
# WHY: Automatically find semantic breakpoints rather than setting
#      one fixed cosine-similarity cutoff.
# NOTE: threshold here is a standard-deviation multiplier, NOT cosine.
# ============================================================
def semantic_chunking(
    text: str | None = None,
    threshold: float = 1.0,
    save_to_chroma: bool = True,
) -> list[str]:
    """Return semantic chunks using a standard-deviation breakpoint."""
    if text is None:
        text = " ".join(DEMO_SENTENCES)
    if not text.strip():
        return []

    # Step 1: Embed sentences to discover distances between their meanings.
    embedding_model = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    splitter = SemanticChunker(
        embeddings=embedding_model,
        breakpoint_threshold_type="standard_deviation",
        breakpoint_threshold_amount=threshold,
        buffer_size=0,
    )

    # Step 2: LangChain decides where to split.
    chunks = splitter.split_text(text)
    print(f"\nLangChain semantic chunks ({len(chunks)}):")
    for number, chunk in enumerate(chunks, start=1):
        print(f"Chunk {number}: {chunk}")

    # Step 3: OPTIONAL vector database persistence.
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


# ============================================================
# 5. MARKDOWN HEADER CHUNKING
# WHAT: Organize text by #, ##, ### headings.
# WHY: Preserve document structure and metadata for retrieval filters.
# RETURNS: list[Document], each with page_content and metadata.
# ============================================================
def markdown_chunking(
    text: str | None = None,
    save_to_chroma: bool = False,
) -> list[Document]:
    """Split Markdown by heading hierarchy and return Documents."""
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

    # Step 1: Tell LangChain which headings become which metadata fields.
    headers = [("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3")]
    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers, strip_headers=True
    )
    # Step 2: Unlike fixed chunking, this returns Document objects.
    documents = splitter.split_text(text)
    for number, document in enumerate(documents, start=1):
        print(f"\nMarkdown chunk {number}")
        print("Content:", document.page_content)
        print("Metadata:", document.metadata)

    # Step 3: OPTIONAL persist documents with their metadata.
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


# ============================================================
# 6. HTML HEADER CHUNKING
# WHAT: Organize HTML by <h1>, <h2>, <h3> headings.
# WHY: Retain HTML page structure as searchable metadata.
# ============================================================
def html_chunking(
    text: str | None = None,
    save_to_chroma: bool = True,
) -> list[Document]:
    """Split HTML by its heading tags and return Documents."""
    if text is None:
        text = """
        <html><body>
          <h1>Database Guide</h1><p>Database troubleshooting guide.</p>
          <h2>Connection Issues</h2><p>Check database connections.</p>
          <h3>Timeout Errors</h3><p>Increase connection timeout.</p>
          <h3>Node Down</h3><p>One WebLogic node is down.</p>
        </body></html>
        """

    # Step 1: Map HTML tag names to metadata keys.
    headers = [("h1", "Header 1"), ("h2", "Header 2"), ("h3", "Header 3")]
    splitter = HTMLHeaderTextSplitter(headers_to_split_on=headers)
    # Step 2: Inspect each Document's content and heading metadata.
    documents = splitter.split_text(text)
    for number, document in enumerate(documents, start=1):
        print(f"\nHTML chunk {number}")
        print("Content:", document.page_content)
        print("Metadata:", document.metadata)

    # Step 3: OPTIONAL persist HTML chunks.
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


# ============================================================
# 7. CHROMA VECTOR STORE + SIMILARITY SEARCH
# WHAT: Embed tickets, store vectors, and search for nearby vectors.
# WHY: RAG retrieves relevant past tickets for a user's question.
# NOTE: A smaller returned *distance* means a closer vector match.
# ============================================================
def chroma_vector_store(
    documents: list[Document] | None = None,
    query: str = "Authentication problems after password reset",
    top_k: int = 3,
) -> Chroma:
    """Build this example's own Chroma collection and query it."""
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
    if top_k < 1:
        raise ValueError("top_k must be at least 1")

    # Step 1: Attach an embedding model to a persistent Chroma collection.
    embedding_model = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    store = Chroma(
        collection_name="whole_documents_demo",
        embedding_function=embedding_model,
        persist_directory=str(DB_DIRECTORY),
    )
    # Step 2: Each document becomes a vector + text + metadata in Chroma.
    ids = [f"whole_ticket_{i}" for i in range(len(documents))]
    store.add_documents(documents, ids=ids)

    # Step 3: Chroma embeds query and searches for closest stored vectors.
    results = store.similarity_search_with_score(query=query, k=top_k)
    for number, (document, distance) in enumerate(results, start=1):
        print(f"\nResult {number}: {document.page_content}")
        print("Metadata:", document.metadata)
        print(f"Distance: {distance:.4f}")
    return store


# ============================================================
# 8. MAXIMUM MARGINAL RELEVANCE (MMR)
# WHAT: Pick documents that are relevant AND not too repetitive.
# WHY: RAG needs diverse evidence, not several near-identical tickets.
# k: final returned count; fetch_k: candidate count.
# lambda_mult=1 prefers relevance; 0 prefers diversity.
# ============================================================
def mmr_search(
    documents: list[Document] | None = None,
    query: str = "Authentication problems after password reset",
    top_k: int = 2,
    fetch_k: int = 3,
    lambda_mult: float = 0.5,
) -> list[Document]:
    """Create an independent collection and run MMR search."""
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
    if not 1 <= top_k <= fetch_k:
        raise ValueError("Require 1 <= top_k <= fetch_k")
    if not 0 <= lambda_mult <= 1:
        raise ValueError("lambda_mult must be between 0 and 1")

    # Step 1: This lesson builds its OWN store, not lesson 7's store.
    embedding_model = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    store = Chroma(
        collection_name="support_tickets_mmr_demo",
        embedding_function=embedding_model,
        persist_directory=str(DB_DIRECTORY),
    )
    store.add_documents(documents, ids=[f"mmr_ticket_{i}" for i in range(len(documents))])

    # Step 2: Fetch up to fetch_k candidates; select top_k for diversity.
    results = store.max_marginal_relevance_search(
        query=query, k=top_k, fetch_k=fetch_k, lambda_mult=lambda_mult
    )
    for number, document in enumerate(results, start=1):
        print(f"\nMMR result {number}: {document.page_content}")
        print("Metadata:", document.metadata)
    return results


# ============================================================
# 9. CHROMA METADATA FILTERING
# WHAT: Require a structured metadata field, e.g. category=Authentication.
# WHY: Search only the relevant subset of support tickets.
# IMPORTANT: Metadata filtering differs from semantic similarity.
# ============================================================
def metadata_filtering(
    documents: list[Document] | None = None,
    query: str = "Login issues after password reset",
    category: str = "Authentication",
    top_k: int = 3,
) -> list[Document]:
    """Compare unfiltered and metadata-filtered semantic search."""
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
    if top_k < 1:
        raise ValueError("top_k must be at least 1")

    # Step 1: Store content, embedding, and each ticket's metadata.
    embedding_model = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    store = Chroma(
        collection_name="support_tickets_metadata_demo",
        embedding_function=embedding_model,
        persist_directory=str(DB_DIRECTORY),
    )
    store.add_documents(documents, ids=[f"metadata_ticket_{i}" for i in range(len(documents))])

    # Step 2: Regular semantic retrieval can return ANY category.
    all_results = store.similarity_search(query=query, k=top_k)
    print("\nWITHOUT metadata filter:")
    for document in all_results:
        print(document.metadata.get("category"), "-", document.page_content)

    # Step 3: Chroma restricts the candidate set to matching metadata.
    filtered_results = store.similarity_search(
        query=query, k=top_k, filter={"category": category}
    )
    print(f"\nWITH category={category!r} filter:")
    for document in filtered_results:
        print(document.metadata.get("category"), "-", document.page_content)
    return filtered_results


# ============================================================
# MAIN: Run one independent example at a time
# 1..9 = one example, "ALL" = all nine, None = no examples.
# DEFAULT 1 avoids unintentional API charges during initial review.
# ============================================================
EXAMPLE_TO_RUN = "ALL"


def main() -> None:
    """Choose and execute examples; the numbered functions stay independent."""
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
            print(f"\n========== EXAMPLE {number}: {example.__name__} ==========")
            example()
    elif EXAMPLE_TO_RUN is not None:
        if EXAMPLE_TO_RUN not in examples:
            raise ValueError('EXAMPLE_TO_RUN must be 1 through 9, "ALL", or None')
        examples[EXAMPLE_TO_RUN]()


if __name__ == "__main__":
    main()
