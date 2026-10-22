
"""
RAG CHUNKING STRATEGIES - LEARNING REFERENCE
============================================

1. Fixed-Size Chunking       - Split by character count
2. Recursive Chunking        - Split using natural text boundaries
3. Manual Semantic Chunking  - Fixed cosine similarity threshold
4. LangChain Semantic        - Statistical embedding breakpoints
5. Markdown Chunking         - Split using Markdown headings

All functions print results and return chunks.
"""

import os
import textwrap
import numpy as np
from pathlib import Path

from langchain_openai import OpenAIEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter
)
from langchain_core.documents import Document
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# CONFIGURATION
# ============================================================

EMBEDDING_MODEL = os.getenv(
    "OPENAI_EMBEDDING_MODEL",
    "text-embedding-3-small"
)

DEMO_SENTENCES = [
    "Users cannot log in.",
    "Password reset is failing.",
    "Authentication service has errors.",
    "Database connections are timing out.",
    "Connection pool is exhausted.",
    "SQL queries are running slowly."
]


# ============================================================
# 1. FIXED-SIZE CHUNKING
# Splits after a fixed number of characters.
# Does not understand words, sentences, or meaning.
# ============================================================

def fixed_size_chunking(
    text: str = "Apple Banana Mango Orange Grapes Pineapple",
    chunk_size: int = 12,
    overlap: int = 2
) -> list[str]:

    splitter = CharacterTextSplitter(
        separator="",
        chunk_size=chunk_size,
        chunk_overlap=overlap
    )

    fixed_chunks = splitter.split_text(text)

    print(f"Total chunks: {len(fixed_chunks)}")
    fixedChunksDocuments = []
    for i, chunk in enumerate(fixed_chunks, 1):
        print(f"Chunk {i}: {fixed_chunks} ({len(fixed_chunks)} chars)")
        doc = Document(  page_content=chunk )
        fixedChunksDocuments.append(doc)
    # Use LangChain's embedding wrapper (handles API calls internally)
    embeddings_model = OpenAIEmbeddings(
        model=os.getenv('OPENAI_EMBEDDING_MODEL', 'text-embedding-3-small')
    )
  
    # Store 2: Fixed-size chunks (may split mid-sentence)

    db_path = Path(__file__).resolve().parent / "chroma_db"
    # Step 3: Create or open Chroma collection
    store_fixed = Chroma.from_documents(
        documents=fixedChunksDocuments,
        embedding=embeddings_model,
        collection_name="fixed_chunks",
        persist_directory=str(db_path)
    )
    return fixed_chunks


# ============================================================
# 2. RECURSIVE CHARACTER SPLITTING
# Tries paragraphs, lines, words, then characters.
# Preserves natural text boundaries when possible.
# ============================================================

def recursive_chunking(
    text: str = "Apple Banana Mango Orange Grapes Pineapple",
    chunk_size: int = 12,
    overlap: int = 0
) -> list[str]:

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", " ", ""]
    )

    recursive_chunks = splitter.split_text(text)

    print(f"Total chunks: {len(recursive_chunks)}")
    recursiveChunksDocuments = []
    for i, chunk in enumerate(recursive_chunks, 1):
        print(f"recursive_chunks {i}: {recursive_chunks} ({len(recursive_chunks)} chars)")
        doc = Document(  page_content=recursive_chunks )
        recursiveChunksDocuments.append(doc)
     # Use LangChain's embedding wrapper (handles API calls internally)
        embeddings_model = OpenAIEmbeddings(
            model=os.getenv('OPENAI_EMBEDDING_MODEL', 'text-embedding-3-small')
        )
      
        # Store 2: Fixed-size chunks (may split mid-sentence)
        store_fixed = Chroma.from_documents(
            documents=recursiveChunksDocuments,
            embedding=embeddings_model,
            collection_name="recursive_chunks"
        )
    return recursive_chunks


# ============================================================
# 3. MANUAL SEMANTIC CHUNKING
# Compare embeddings of consecutive sentences.
# Split when cosine similarity < threshold.
# ============================================================

def manual_semantic_chunking(
    sentences: list[str] | None = None,
    threshold: float = 0.45
) -> list[str]:

    if sentences is None:
        sentences = DEMO_SENTENCES

    if not sentences:
        return []

    model = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    vectors = np.array(model.embed_documents(sentences))

    chunks = []
    current_chunk = [sentences[0]]

    for i in range(len(sentences) - 1):

        score = cosine_similarity(
            vectors[i:i+1],
            vectors[i+1:i+2]
        )[0][0]

        if score < threshold:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentences[i+1]]
        else:
            current_chunk.append(sentences[i+1])

    chunks.append(" ".join(current_chunk))

    print(f"Total chunks: {len(chunks)}")

    for i, chunk in enumerate(chunks, 1):
        print(f"\nChunk {i}: {chunk}")

    return chunks


# ============================================================
# 4. LANGCHAIN SEMANTIC CHUNKING
# Uses embeddings and statistical cosine distance.
# buffer_size=0 compares individual sentences.
# ============================================================

def semantic_chunking(
    text: str | None = None,
    threshold: float = 1.0
) -> list[str]:

    if text is None:
        text = " ".join(DEMO_SENTENCES)

    if not text.strip():
        return []

    model = OpenAIEmbeddings(model=EMBEDDING_MODEL)

    splitter = SemanticChunker(
        embeddings=model,
        breakpoint_threshold_type="standard_deviation",
        breakpoint_threshold_amount=threshold,
        buffer_size=0
    )

    Semanticchunks = splitter.split_text(text)

    print(f"Total chunks: {len(Semanticchunks)}")

    for i, chunk in enumerate(Semanticchunks, 1):
        print(f"\nChunk {i}: {Semanticchunks}")
    SemanticchunksDocuments = []
    for i, chunk in enumerate(Semanticchunks, 1):
        print(f"Semanticchunks {i}: {Semanticchunks} ({len(Semanticchunks)} chars)")
        doc = Document(  page_content=Semanticchunks )
        SemanticchunksDocuments.append(doc)
        # Use LangChain's embedding wrapper (handles API calls internally)
        embeddings_model = OpenAIEmbeddings(
            model=os.getenv('OPENAI_EMBEDDING_MODEL', 'text-embedding-3-small')
        )
        
        # Store 2: Fixed-size chunks (may split mid-sentence)
        store_fixed = Chroma.from_documents(
            documents=recursiveChunksDocuments,
            embedding=embeddings_model,
            collection_name="recursive_chunks"
        )
    return chunks


# ============================================================
# 5. MARKDOWN HEADER CHUNKING
# Splits based on #, ##, ### headers.
# Each Document contains page_content and metadata.
# ============================================================

def markdown_chunking(
    text: str | None = None
) -> list[Document]:

    if text is None:
        text = textwrap.dedent("""
            # Database
            Database troubleshooting guide.

            ## Connection
            Check database connections.

            ### Timeout
            Increase connection timeout.

            ### Node Down
            One WebLogic node is down.
        """)

    headers = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3")
    ]

    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers,
        strip_headers=False
    )

    markdownchunks = splitter.split_text(text)

    print(f"Total chunks: {len(markdownchunks)}")

    for i, chunk in enumerate(markdownchunks, 1):
        print(f"\markdownchunks {i}")
        print("Content:", chunk.page_content)
        print("Metadata:", chunk.metadata)

    embeddings_model = OpenAIEmbeddings(
    model=os.getenv('OPENAI_EMBEDDING_MODEL', 'text-embedding-3-small')
    )

    # Store 1: markdownchunks 
    store_whole = Chroma.from_documents(
    documents=markdownchunks,
    embedding=embeddings_model,
    collection_name="whole_docs"
    )
    return markdownchunks

from langchain_text_splitters import HTMLHeaderTextSplitter
from langchain_core.documents import Document


# ============================================================
# 6. HTML HEADER CHUNKING
# Split by <h1>, <h2>, <h3> tags.
# Preserve heading hierarchy in Document metadata.
# No embeddings required.
# ============================================================

def html_chunking(text: str | None = None) -> list[Document]:

    # Step 1: Default HTML for learning
    if text is None:
        text = """
        <html>
        <body>
            <h1>Database Guide</h1>
            <p>Database troubleshooting guide.</p>

            <h2>Connection Issues</h2>
            <p>Check database connections.</p>

            <h3>Timeout Errors</h3>
            <p>Increase connection timeout.</p>

            <h3>Node Down</h3>
            <p>One WebLogic node is down.</p>
            
        </body>
        </html>
        """

    # Step 2: Define HTML headers
    headers = [
        ("h1", "Header 1"),
        ("h2", "Header 2"),
        ("h3", "Header 3")
    ]

    # Step 3: Initialize HTML splitter
    splitter = HTMLHeaderTextSplitter(
        headers_to_split_on=headers
    )

    # Step 4: Split HTML into Document objects
    htmlHeaderchunks = splitter.split_text(text)

    # Step 5: Print chunks for learning
    print(f"\nTotal HTML chunks: {len(htmlHeaderchunks)}")

    for i, chunk in enumerate(htmlHeaderchunks, 1):
        print(f"\nChunk {i}")
        print("Content:", chunk.page_content)
        print("Metadata:", chunk.metadata)

    # Step 6: Return for reuse
    embeddings_model = OpenAIEmbeddings(
    model=os.getenv('OPENAI_EMBEDDING_MODEL', 'text-embedding-3-small')
    )

    # Store 1: markdownchunks 
    store_whole = Chroma.from_documents(
    documents=htmlHeaderchunks,
    embedding=embeddings_model,
    collection_name="HtmlHeaderchunks"
    )
    return htmlHeaderchunks


from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document



# ============================================================
# 7. CHROMA VECTOR STORE
# Store text + embeddings + metadata and perform similarity search.
# Data is persisted locally in ./chroma_db.
# ============================================================

def chroma_vector_store(
    documents: list[Document] | None = None,
    query: str = "Authentication problems after password reset",
    top_k: int = 3
) -> Chroma:

    # Step 1: Default sample documents
    if documents is None:
        documents = [
            Document(
                page_content="Users cannot log in after password reset.",
                metadata={"ticket_id": "TICK-001", "category": "Authentication"}
            ),
            Document(
                page_content="Database connection timeout in production.",
                metadata={"ticket_id": "TICK-002", "category": "Database"}
            ),
            Document(
                page_content="Password reset link expired.",
                metadata={"ticket_id": "TICK-003", "category": "Authentication"}
            )
        ]

    if not documents:
        raise ValueError("documents cannot be empty")

    # Step 2: Initialize embedding model
    model = OpenAIEmbeddings(model="text-embedding-3-small")
    
    
    db_path = Path(__file__).resolve().parent / "chroma_db"
    # Step 3: Create or open Chroma collection
    store = Chroma(
        collection_name="WholeDocuments",
        embedding_function=model,
        persist_directory=str(db_path)
    )

    # Step 4: Store documents using stable IDs
    # Existing documents with these IDs are updated on reruns.
    ids = [f"ticket_{i}" for i in range(len(documents))]

    store.add_documents(
        documents=documents,
        ids=ids
    )

    print(f"\nStored {len(documents)} documents in ChromaDB")

    # Step 5: Search similar documents
    results = store.similarity_search_with_score(
        query=query,
        k=top_k
    )

    # Step 6: Display results
    print(f"\nQuery: {query}")
    print(f"Total matches: {len(results)}")

    for i, (doc, distance) in enumerate(results, 1):
        print(f"\nResult {i}")
        print("Content:", doc.page_content)
        print("Metadata:", doc.metadata)
        print(f"Distance: {distance:.4f}")






# ============================================================
# 8. MMR SEARCH (MAXIMAL MARGINAL RELEVANCE)
# ============================================================
#
# MMR SEARCH PARAMETERS
#
# k=2:
#   Return up to 2 final documents.
#
# fetch_k=3:
#   Retrieve up to 3 similar candidate documents.
#   MMR evaluates these candidates before selecting k.
#
# lambda_mult=0.5:
#   Controls the balance of relevance and diversity.
#
#   1.0 -> Prioritize relevance (similarity search behavior)
#   0.5 -> Balance relevance and diversity
#   0.0 -> Prioritize diversity after the first selection
#
# MMR first selects the most relevant document.
# Then it selects additional documents that balance
# query relevance with diversity from selected results.
#
# NOTE:
# Key takeaway: fetch_k determines how many documents MMR can choose from, 
# k determines how many it returns, and lambda_mult controls how it chooses them.
# MMR does not create embeddings differently.
# It changes how documents are selected during retrieval.
# ============================================================

def mmr_search(
    documents: list[Document] | None = None,
    query: str = "Authentication problems after password reset",
    top_k: int = 2
) -> list[Document]:

    # Step 1: Default sample documents
    if documents is None:
        documents = [
            Document(
                page_content="Users cannot log in after password reset.",
                metadata={
                    "ticket_id": "TICK-001",
                    "category": "Authentication"
                }
            ),
            Document(
                page_content="Database connection timeout in production.",
                metadata={
                    "ticket_id": "TICK-002",
                    "category": "Database"
                }
            ),
            Document(
                page_content="Password reset link expired.",
                metadata={
                    "ticket_id": "TICK-003",
                    "category": "Authentication"
                }
            )
        ]

    if not documents:
        raise ValueError("documents cannot be empty")

    # Step 2: Initialize embedding model
    # Each document is represented by a 1536-dimensional vector.
    model = OpenAIEmbeddings(model="text-embedding-3-small")

    # Step 3: Create or open a separate MMR collection
    # Use the script directory instead of the terminal directory.
    db_path = Path(__file__).resolve().parent / "chroma_db"

    store = Chroma(
        collection_name="support_tickets_MMR_demo",
        embedding_function=model,
        persist_directory=str(db_path)
    )

    # Step 4: Store documents using stable IDs
    # Reusing IDs updates existing documents instead of duplicating them.
    ids = [f"ticket_{i}" for i in range(len(documents))]

    store.add_documents(
        documents=documents,
        ids=ids
    )

    print(f"\nStored {len(documents)} documents in ChromaDB")

    # Step 5: Perform MMR search
    # fetch_k = number of candidates considered before MMR selection
    # k = number of final documents returned
    # lambda_mult = relevance/diversity tradeoff
    mmr_results = store.max_marginal_relevance_search(
        query=query,
        k=top_k,
        fetch_k=3,
        lambda_mult=0.5
    )

    # Step 6: Display search results
    print("\n===== MMR SEARCH RESULTS =====")
    print(f"Query: {query}")
    print(f"Total matches: {len(mmr_results)}")

    for i, doc in enumerate(mmr_results, 1):
        print(f"\nResult {i}")
        print("Content:", doc.page_content)
        print("Ticket ID:", doc.metadata.get("ticket_id"))
        print("Category:", doc.metadata.get("category"))



# ============================================================
# 9. METADATA FILTERING IN CHROMADB
# ============================================================
#
# PURPOSE:
# Search documents based on BOTH:
#   1. Semantic similarity (meaning of the query)
#   2. Metadata filtering (structured conditions)
#
# WITHOUT FILTERING:
#   Search across ALL support tickets.
#
# WITH FILTERING:
#   Search ONLY tickets matching specific metadata.
#
# EXAMPLE:
# Query: "Login issues after password reset"
#
# Without filtering:
#   ChromaDB searches Authentication, Database, Payment, etc.
#
# With filter={"category": "Authentication"}:
#   ChromaDB searches ONLY Authentication tickets.
#
# HOW IT WORKS:
#   Step 1: Create sample documents with metadata.
#   Step 2: Convert documents into embeddings.
#   Step 3: Store documents and metadata in ChromaDB.
#   Step 4: Apply metadata filter before similarity search.
#   Step 5: Return the most similar matching documents.
#
# USE CASES:
#   - Search tickets from a specific category.
#   - Search only HIGH priority incidents.
#   - Search tickets for a particular customer.
#   - Search tickets within a date range.
#
# IMPORTANT:
# Metadata filtering checks structured values.
# Semantic similarity compares vector embeddings.
# ============================================================

def metadata_filtering(
    documents: list[Document] | None = None,
    query: str = "Login issues after password reset",
    category: str = "Authentication",
    top_k: int = 3
) -> list[Document]:

    # Step 1: Create sample documents with metadata
    if documents is None:
        documents = [
            Document(
                page_content="Users cannot log in after password reset.",
                metadata={
                    "ticket_id": "TICK-001",
                    "category": "Authentication",
                    "priority": "High"
                }
            ),
            Document(
                page_content="Database connection timeout in production.",
                metadata={
                    "ticket_id": "TICK-002",
                    "category": "Database",
                    "priority": "High"
                }
            ),
            Document(
                page_content="Password reset link expired.",
                metadata={
                    "ticket_id": "TICK-003",
                    "category": "Authentication",
                    "priority": "Medium"
                }
            ),
            Document(
                page_content="User cannot log in because MFA token expired.",
                metadata={
                    "ticket_id": "TICK-004",
                    "category": "Authentication",
                    "priority": "High"
                }
            )
        ]

    if not documents:
        raise ValueError("documents cannot be empty")

    # Step 2: Initialize embedding model
    # Each document becomes a 1536-dimensional vector.
    model = OpenAIEmbeddings(model="text-embedding-3-small")

    # Step 3: Create or open ChromaDB collection
    # Use a separate collection for this learning example.
    db_path = Path(__file__).resolve().parent / "chroma_db"

    store = Chroma(
        collection_name="support_tickets_metadata_demo",
        embedding_function=model,
        persist_directory=str(db_path)
    )

    # Step 4: Store documents with stable IDs
    # Reusing IDs updates existing records on reruns.
    ids = [f"metadata_ticket_{i}" for i in range(len(documents))]

    store.add_documents(
        documents=documents,
        ids=ids
    )

    print(f"\nStored {len(documents)} documents in ChromaDB")

    # Step 5: Perform semantic search WITHOUT metadata filtering
    print("\n===== SEARCH WITHOUT FILTER =====")

    all_results = store.similarity_search(
        query=query,
        k=top_k
    )

    for i, doc in enumerate(all_results, 1):
        print(f"\nResult {i}")
        print("Ticket:", doc.metadata.get("ticket_id"))
        print("Category:", doc.metadata.get("category"))
        print("Content:", doc.page_content)

    # Step 6: Perform semantic search WITH metadata filtering
    # Only documents where category == "Authentication"
    # are eligible for retrieval.
    print(f"\n===== SEARCH WITH FILTER: {category} =====")

    filtered_results = store.similarity_search(
        query=query,
        k=top_k,
        filter={"category": category}
    )

    # Step 7: Display filtered results
    print(f"\nFiltered results: {len(filtered_results)}")

    for i, doc in enumerate(filtered_results, 1):
        print(f"\nResult {i}")
        print("Ticket:", doc.metadata.get("ticket_id"))
        print("Category:", doc.metadata.get("category"))
        print("Priority:", doc.metadata.get("priority"))
        print("Content:", doc.page_content)

    # Step 8: Return filtered documents for future RAG use
    return filtered_results


# ============================================================
# MAIN - CALL ALL FIVE STRATEGIES
# ============================================================

def main():

    print("\n===== 1. FIXED-SIZE CHUNKING =====")
    fixed_size_chunking()

    print("\n===== 2. RECURSIVE CHARACTER SPLITTING =====")
    recursive_chunking()

    print("\n===== 3. MANUAL SEMANTIC CHUNKING =====")
    manual_semantic_chunking()

    print("\n===== 4. LANGCHAIN SEMANTIC CHUNKING =====")
    semantic_chunking()

    print("\n===== 5. MARKDOWN HEADER CHUNKING =====")
    markdown_chunking()

    print("\n===== 5. MARKDOWN HEADER CHUNKING =====")
    html_chunking()

    print("\n===== 7. Creates or updates documents in ChromaDB =====")   
    chroma_vector_store()


    print("\n===== 8: Opens ChromaDB independently and performs MMR =====")   
    mmr_search()

    print("\n===== 9. METADATA FILTERING =====")
    metadata_filtering()

if __name__ == "__main__":
    main()


