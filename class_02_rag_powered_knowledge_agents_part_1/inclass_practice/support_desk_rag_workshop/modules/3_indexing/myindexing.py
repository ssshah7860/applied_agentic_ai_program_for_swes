"""CLASS 02 - MODULE 3: INDEXING LEARNING REFERENCE

Each example is STANDALONE: it creates its own documents, indexes and models.
Change EXAMPLE_TO_RUN to a number (1-19), "ALL", or None.

Install (Python 3.11 recommended):
    python -m pip install llama-index-core llama-index-embeddings-openai \
        llama-index-llms-openai llama-index-vector-stores-chroma \
        chromadb python-dotenv

Set OPENAI_API_KEY in your environment or a .env file beside this script.
Some examples call paid OpenAI APIs. Tree and Keyword builds can be slow.
Examples 7, 8 and 9 create local persistence folders.
"""

from pathlib import Path
from dotenv import load_dotenv
import os
root_env = Path(__file__).resolve().parents[5] / ".env"

# Load environment variables
load_dotenv(root_env)
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("OPENAI_API_KEY is missing")

# Select 1-19, "ALL", or None. ALL makes many API calls.
EXAMPLE_TO_RUN = "ALL"


# 1. DOCUMENTS, NODES, AND INDEX
# Concept: Documents are inputs; nodes are searchable chunks; index is NOT a node.
def example_01_documents_nodes_index():
    from llama_index.core import Document, VectorStoreIndex, Settings
    from llama_index.core.node_parser import SentenceSplitter
    from llama_index.embeddings.openai import OpenAIEmbedding

    documents = [Document(text=("Password reset failed. " * 15),
                          metadata={"ticket_id": "TICK-001"})]
    splitter = SentenceSplitter(chunk_size=64, chunk_overlap=0)
    nodes = splitter.get_nodes_from_documents(documents)
    Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
    index = VectorStoreIndex(nodes)
    print("Documents:", len(documents), "Nodes:", len(nodes))
    for node in nodes:
        print("Node:", node.node_id[:8], repr(node.text[:65]))
    print("Index object type:", type(index).__name__)


# 2. VECTOR INDEX / SEMANTIC RETRIEVAL
# Concept: Embeddings let a vector index find related meanings.
def example_02_vector_index():
    from llama_index.core import Document, VectorStoreIndex, Settings
    from llama_index.embeddings.openai import OpenAIEmbedding

    Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
    documents = [
        Document(text="Password reset causes login failure", metadata={"ticket_id": "T1"}),
        Document(text="Database connection timeout", metadata={"ticket_id": "T2"}),
        Document(text="Payment gateway error", metadata={"ticket_id": "T3"}),
    ]
    index = VectorStoreIndex.from_documents(documents)
    results = index.as_retriever(similarity_top_k=2).retrieve("cannot sign in")
    for result in results:
        print(result.metadata["ticket_id"], round(result.score, 3), result.text)


# 3. RETRIEVER VS QUERY ENGINE
# Concept: Retriever returns nodes; query engine also generates an LLM answer.
def example_03_retriever_vs_query_engine():
    from llama_index.core import Document, VectorStoreIndex, Settings
    from llama_index.embeddings.openai import OpenAIEmbedding
    from llama_index.llms.openai import OpenAI

    Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
    Settings.llm = OpenAI(model="gpt-4o-mini")
    documents = [Document(text="For login failures after password reset, verify the new password.")]
    index = VectorStoreIndex.from_documents(documents)
    question = "What should I check after resetting my password?"
    print("RETRIEVER (no answer-generation LLM):")
    for item in index.as_retriever(similarity_top_k=1).retrieve(question):
        print(item.text)
    print("QUERY ENGINE (generates answer):")
    print(index.as_query_engine(similarity_top_k=1).query(question).response)


# 4. SUMMARY INDEX
# Concept: Stores ordered nodes; by default query retrieves all for synthesis.
def example_04_summary_index():
    from llama_index.core import Document, SummaryIndex, Settings
    from llama_index.llms.openai import OpenAI

    Settings.llm = OpenAI(model="gpt-4o-mini")
    documents = [Document(text="Login issue: expired password."),
                 Document(text="Database issue: connection pool exhausted."),
                 Document(text="Billing issue: international card declined.")]
    index = SummaryIndex.from_documents(documents)
    print("Retrieved node count:", len(index.as_retriever().retrieve("common issues")))
    response = index.as_query_engine(response_mode="tree_summarize").query(
        "Summarize the kinds of issues in these tickets."
    )
    print("Answer:", response.response)


# 5. TREE INDEX
# Concept: num_children builds hierarchy; child_branch_factor guides traversal.
def example_05_tree_index():
    from llama_index.core import Document, TreeIndex, Settings
    from llama_index.llms.openai import OpenAI

    Settings.llm = OpenAI(model="gpt-4o-mini")
    documents = [Document(text=t) for t in [
        "Password reset failure", "SSO login failure", "MFA token expired",
        "Database timeout", "Slow SQL query", "Connection pool exhausted",
    ]]
    index = TreeIndex.from_documents(documents, num_children=2)
    print("num_children=2: up to two children per summary group")
    engine = index.as_query_engine(child_branch_factor=1)
    print("child_branch_factor=1: follow a narrow path")
    print("Answer:", engine.query("What authentication issues occurred?").response)


# 6. KEYWORD TABLE INDEX
# Concept: Keyword -> node IDs, no embedding vectors needed.
def example_06_keyword_index():
    from llama_index.core import Document, KeywordTableIndex, Settings
    from llama_index.llms.openai import OpenAI

    Settings.llm = OpenAI(model="gpt-4o-mini")
    documents = [Document(text="Password reset failure and login error"),
                 Document(text="Database connection timeout"),
                 Document(text="Payment processing failure")]
    index = KeywordTableIndex.from_documents(documents)
    print("Extracted keywords:", sorted(index.index_struct.table.keys()))
    print("Keyword retrieval:")
    for item in index.as_retriever().retrieve("password"):
        print(item.text)
    # LLM keyword extraction may omit exact IDs; use metadata filters for strict ID matching.


# 7. HYBRID RETRIEVAL + SIMPLE FUSION
# Concept: Combine vector and keyword results; keep first occurrence of each node.
def example_07_hybrid_simple_fusion():
    from llama_index.core import Document, VectorStoreIndex, KeywordTableIndex, Settings
    from llama_index.embeddings.openai import OpenAIEmbedding
    from llama_index.llms.openai import OpenAI

    Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
    Settings.llm = OpenAI(model="gpt-4o-mini")
    documents = [
        Document(text="Password reset login failure", metadata={"ticket_id": "T1"}),
        Document(text="SSO authentication failure", metadata={"ticket_id": "T2"}),
        Document(text="Database timeout", metadata={"ticket_id": "T3"}),
    ]
    vector_index = VectorStoreIndex.from_documents(documents)
    keyword_index = KeywordTableIndex.from_documents(documents)
    query = "authentication login"
    vector_results = vector_index.as_retriever(similarity_top_k=2).retrieve(query)
    keyword_results = keyword_index.as_retriever().retrieve(query)
    seen, combined = set(), []
    for item in vector_results + keyword_results:
        ticket_id = item.metadata["ticket_id"]
        if ticket_id not in seen:
            seen.add(ticket_id)
            combined.append(ticket_id)
    print("Vector:", [x.metadata["ticket_id"] for x in vector_results])
    print("Keyword:", [x.metadata["ticket_id"] for x in keyword_results])
    print("Simple fusion (no reranking):", combined)


# 8. RECIPROCAL RANK FUSION (RRF)
# Concept: Combine ranking positions, not similarity scores. Pure Python.
def example_08_rrf():
    vector_results = ["T1", "T2", "T3"]
    keyword_results = ["T3", "T4"]
    k = 60
    scores = {}
    for ranking in [vector_results, keyword_results]:
        for rank, ticket_id in enumerate(ranking, start=1):
            scores[ticket_id] = scores.get(ticket_id, 0) + 1 / (k + rank)
    for ticket_id, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        print(ticket_id, f"RRF score={score:.6f}")


# 9. METADATA FILTERING
# Concept: Filter eligible nodes before vector similarity search.
def example_09_metadata_filtering():
    from llama_index.core import Document, VectorStoreIndex, Settings
    from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter
    from llama_index.embeddings.openai import OpenAIEmbedding

    Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
    documents = [
        Document(text="Password reset failed", metadata={"category": "Authentication", "priority": "High"}),
        Document(text="SSO login failed", metadata={"category": "Authentication", "priority": "Low"}),
        Document(text="Database timeout", metadata={"category": "Database", "priority": "High"}),
    ]
    index = VectorStoreIndex.from_documents(documents)
    filters = MetadataFilters(filters=[
        ExactMatchFilter(key="category", value="Authentication"),
        ExactMatchFilter(key="priority", value="High"),
    ])
    results = index.as_retriever(similarity_top_k=3, filters=filters).retrieve("login issues")
    for item in results:
        print(item.text, item.metadata)


# 10. LLAMAINDEX LOCAL PERSISTENCE
# Concept: Save default in-memory vector index and reload without re-embedding docs.
def example_10_llamaindex_persistence():
    from tempfile import TemporaryDirectory
    from llama_index.core import Document, VectorStoreIndex, StorageContext, load_index_from_storage, Settings
    from llama_index.embeddings.openai import OpenAIEmbedding

    Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
    documents = [Document(text="Password reset failure"), Document(text="Database timeout")]
    with TemporaryDirectory(prefix="llama_index_demo_") as folder:
        index = VectorStoreIndex.from_documents(documents)
        index.storage_context.persist(persist_dir=folder)
        print("SAVED to:", folder)
        storage_context = StorageContext.from_defaults(persist_dir=folder)
        loaded = load_index_from_storage(storage_context)
        print("LOADED:", [x.text for x in loaded.as_retriever(similarity_top_k=1).retrieve("password")])
    print("Temporary example files cleaned up; use a permanent folder in a real project.")


# 11. CHROMA PERSISTENCE WITHOUT LLAMAINDEX
# Concept: Chroma embedding function + PersistentClient; no LlamaIndex classes.
def example_11_chroma_only_persistence():
    import os
    import chromadb
    from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

    folder = Path(__file__).with_name("indexing_chroma_only_db")
    print(f"----------{folder}")
    client = chromadb.PersistentClient(path=str(folder))
    embedder = OpenAIEmbeddingFunction(
        api_key=os.environ["OPENAI_API_KEY"], model_name="text-embedding-3-small"
    )
    collection = client.get_or_create_collection("tickets", embedding_function=embedder)
    collection.upsert(ids=["T1", "T2"],
                      documents=["Password reset failure", "Database timeout"])
    print("SAVED records:", collection.count())
    reopened = chromadb.PersistentClient(path=str(folder)).get_collection(
        "tickets", embedding_function=embedder
    )
    result = reopened.query(query_texts=["login password"], n_results=1)
    print("REOPENED result:", result["documents"][0])


# 12. CHROMA PERSISTENCE WITH LLAMAINDEX
# Concept: LlamaIndex embeds nodes; Chroma stores/searches persistent vectors.
def example_12_llamaindex_chroma_persistence():
    import chromadb
    from llama_index.core import Document, VectorStoreIndex, StorageContext, Settings
    from llama_index.embeddings.openai import OpenAIEmbedding
    from llama_index.vector_stores.chroma import ChromaVectorStore

    Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
    folder = Path(__file__).with_name("indexing_llama_chroma_db")
    client = chromadb.PersistentClient(path=str(folder))
    collection = client.get_or_create_collection("tickets")
    store = ChromaVectorStore(chroma_collection=collection)
    # Only insert if empty, so repeated runs do not create duplicate nodes.
    if collection.count() == 0:
        documents = [Document(text="Password reset failure"),
                     Document(text="Database timeout")]
        VectorStoreIndex.from_documents(
            documents, storage_context=StorageContext.from_defaults(vector_store=store)
        )
    print("SAVED Chroma records:", collection.count())
    reopened_collection = chromadb.PersistentClient(path=str(folder)).get_collection("tickets")
    reopened_store = ChromaVectorStore(chroma_collection=reopened_collection)
    loaded = VectorStoreIndex.from_vector_store(reopened_store)
    print("RECONNECTED:", [x.text for x in loaded.as_retriever(similarity_top_k=1).retrieve("password")])


# 13. EMBEDDING BATCH SIZE VS INSERT BATCH SIZE
# Concept: Embedding requests group texts; index insertion groups nodes.
def example_13_batching():
    from llama_index.core import Document, VectorStoreIndex, Settings
    from llama_index.embeddings.openai import OpenAIEmbedding

    Settings.embed_model = OpenAIEmbedding(
        model="text-embedding-3-small", embed_batch_size=2
    )
    documents = [Document(text=f"Ticket {i}: database timeout example {i}") for i in range(1, 11)]
    index = VectorStoreIndex.from_documents(documents, insert_batch_size=4)
    print("10 documents -> 10 short nodes (typically)")
    print("embed_batch_size=2: up to 2 texts per embedding API batch")
    print("insert_batch_size=4: up to 4 nodes per insertion batch")
    print("Index:", type(index).__name__)


# 14. MULTI-INDEX ARCHITECTURE + DYNAMIC SELECTION
# Concept: Building multiple indexes differs from routing a query to one.
def example_14_multi_index_routing():
    from llama_index.core import Document, VectorStoreIndex, KeywordTableIndex, Settings
    from llama_index.embeddings.openai import OpenAIEmbedding
    from llama_index.llms.openai import OpenAI

    Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
    Settings.llm = OpenAI(model="gpt-4o-mini")
    indexes = {
        "tickets": VectorStoreIndex.from_documents([
            Document(text="Password reset caused login failure")
        ]),
        "errors": KeywordTableIndex.from_documents([
            Document(text="ORA-12541 means TNS no listener")
        ]),
    }
    query = "What does ORA-12541 mean?"
    selected_name = "errors" if "ORA-" in query.upper() else "tickets"
    selected_index = indexes[selected_name]
    print("MULTI-INDEX: available indexes:", list(indexes))
    print("DYNAMIC SELECTION: chosen index:", selected_name)
    print("Answer:", selected_index.as_query_engine().query(query).response)


# 15. PRECISION, RECALL, AND LATENCY
# Concept: Evaluate retrieved IDs against known relevant IDs; time retrieval.
def example_15_evaluation():
    import time
    from llama_index.core import Document, VectorStoreIndex, Settings
    from llama_index.embeddings.openai import OpenAIEmbedding

    Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
    documents = [
        Document(text="Password reset failure", metadata={"ticket_id": "T1"}),
        Document(text="SSO authentication issue", metadata={"ticket_id": "T2"}),
        Document(text="Database timeout", metadata={"ticket_id": "T3"}),
        Document(text="Account locked", metadata={"ticket_id": "T4"}),
    ]
    index = VectorStoreIndex.from_documents(documents)
    relevant_ids = {"T1", "T2", "T4"}  # Example ground-truth labels
    start = time.perf_counter()
    results = index.as_retriever(similarity_top_k=3).retrieve("authentication login issue")
    latency = time.perf_counter() - start
    retrieved_ids = [r.metadata["ticket_id"] for r in results]
    relevant_retrieved = len(set(retrieved_ids) & relevant_ids)
    precision = relevant_retrieved / len(retrieved_ids) if retrieved_ids else 0
    recall = relevant_retrieved / len(relevant_ids) if relevant_ids else 0
    print("Retrieved:", retrieved_ids, "Relevant:", sorted(relevant_ids))
    print(f"Precision@3={precision:.2f} Recall@3={recall:.2f} Retrieval latency={latency:.3f}s")



# 16. MMR (DIVERSITY)
# Concept: MMR balances relevance with avoiding repetitive results.
def example_16_mmr():
    from llama_index.core import Document, VectorStoreIndex, Settings
    from llama_index.embeddings.openai import OpenAIEmbedding

    Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
    documents = [Document(text=t) for t in [
        "Password reset login failure", "Password reset email failure",
        "SSO authentication failure", "MFA token expired",
    ]]
    index = VectorStoreIndex.from_documents(documents)
    print("Regular top-2 similarity:")
    for item in index.as_retriever(similarity_top_k=2).retrieve("authentication login"):
        print(" ", item.text)
    # MMR support depends on the vector store. LlamaIndex's default simple store
    # may not implement MMR; use Chroma's native MMR-capable integration instead.
    print("MMR: select relevant results while reducing repeated topics.")
    print("For a runnable MMR example, see mychunking.py -> mmr_search().")


# 17. RESPONSE MODES + TOP-K
# Concept: Top-K controls retrieved nodes; response_mode controls LLM synthesis.
def example_17_response_modes_top_k():
    from llama_index.core import Document, VectorStoreIndex, Settings
    from llama_index.embeddings.openai import OpenAIEmbedding
    from llama_index.llms.openai import OpenAI

    Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
    Settings.llm = OpenAI(model="gpt-4o-mini")
    documents = [Document(text="Password reset failed"),
                 Document(text="Account locked after login attempts"),
                 Document(text="Database timeout")]
    index = VectorStoreIndex.from_documents(documents)
    query = "What login issues occurred?"
    for mode in ["compact", "tree_summarize"]:
        engine = index.as_query_engine(similarity_top_k=2, response_mode=mode)
        response = engine.query(query)
        print(f"Mode={mode}; retrieved={len(response.source_nodes)}")
        print("Answer:", response.response)


# 18. QUERY RESULT CACHING
# Concept: lru_cache avoids repeating identical calls within the same process.
def example_18_query_caching():
    from functools import lru_cache

    calls = {"count": 0}

    @lru_cache(maxsize=10)
    def cached_query(question):
        calls["count"] += 1
        return f"Example answer to: {question}"

    print(cached_query("How to reset password?"))
    print(cached_query("How to reset password?"))
    print("Actual function executions:", calls["count"])
    print("This uses a dummy answer to demonstrate caching without API charges.")


# 19. INDEX COMPOSITION (CONCEPTUAL)
# Concept: A parent index can organize descriptions of multiple child indexes.
def example_19_index_composition():
    # Pure Python analogy; not a ComposableGraph API call.
    child_indexes = {
        "authentication": {"type": "VectorStoreIndex", "summary": "Login and password tickets"},
        "performance": {"type": "VectorStoreIndex", "summary": "Database timeouts"},
        "api": {"type": "KeywordTableIndex", "summary": "Error codes and API names"},
    }
    print("PARENT: descriptions of child indexes")
    for name, child in child_indexes.items():
        print(f"  {name}: {child['type']} - {child['summary']}")
    print("ComposableGraph can connect real child indexes under a parent index.")
    print("This simple analogy intentionally avoids version-specific graph APIs.")


def main():
    """Run a single independent concept or every concept."""
    examples = {
        1: example_01_documents_nodes_index,
        2: example_02_vector_index,
        3: example_03_retriever_vs_query_engine,
        4: example_04_summary_index,
        5: example_05_tree_index,
        6: example_06_keyword_index,
        7: example_07_hybrid_simple_fusion,
        8: example_08_rrf,
        9: example_09_metadata_filtering,
        10: example_10_llamaindex_persistence,
        11: example_11_chroma_only_persistence,
        12: example_12_llamaindex_chroma_persistence,
        13: example_13_batching,
        14: example_14_multi_index_routing,
        15: example_15_evaluation,
        16: example_16_mmr,
        17: example_17_response_modes_top_k,
        18: example_18_query_caching,
        19: example_19_index_composition,
    }
    if isinstance(EXAMPLE_TO_RUN, str) and EXAMPLE_TO_RUN.upper() == "ALL":
        selected = examples.items()
    elif EXAMPLE_TO_RUN is None:
        print("Set EXAMPLE_TO_RUN to 1-15 or 'ALL'.")
        return
    elif EXAMPLE_TO_RUN in examples:
        selected = [(EXAMPLE_TO_RUN, examples[EXAMPLE_TO_RUN])]
    else:
        raise ValueError("EXAMPLE_TO_RUN must be 1-19, 'ALL', or None")

    for number, example in selected:
        print(f"\n{'=' * 65}\nEXAMPLE {number}: {example.__name__}\n{'=' * 65}")
        example()


if __name__ == "__main__":
    main()
