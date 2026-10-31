# Workshop Agenda - Day 1
**Date:** February 10, 2026  
**Modules:** 1-3 (Embeddings, Chunking, Indexing)

**Frameworks Covered:**
- 🦜 **LangChain** - Module 2 (Chunking & Vector Stores)
- 🦙 **LlamaIndex** - Module 3 (Indexing Strategies)

---

## Opening: Why RAG?

### The LLM Customization Spectrum

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    TECHNIQUES TO WORK WITH LLMs                             │
└─────────────────────────────────────────────────────────────────────────────┘

  SIMPLE ◀──────────────────────────────────────────────────────▶ COMPLEX
  CHEAP                                                            EXPENSIVE

  ┌──────────────┐      ┌──────────────┐      ┌──────────────┐
  │   PROMPT     │      │     RAG      │      │  FINE-TUNING │
  │ ENGINEERING  │      │              │      │              │
  └──────────────┘      └──────────────┘      └──────────────┘
        │                     │                     │
        ▼                     ▼                     ▼
   "Just ask the         "Give the LLM         "Train the LLM
    LLM nicely"           the right docs"       on your data"
```

### Why Prompt Engineering Isn't Enough

| Limitation | Example |
|------------|---------|
| **Knowledge cutoff** | LLM doesn't know about events after training |
| **No private data** | Can't answer "What's in our internal docs?" |
| **Hallucination risk** | Makes up facts when it doesn't know |
| **Context window limits** | Can't paste your entire knowledge base |

### Why Not Jump to Fine-Tuning?

| Challenge | Impact |
|-----------|--------|
| **Expensive** | Training costs, GPU compute |
| **Slow iteration** | Days/weeks to retrain |
| **Data requirements** | Need curated training datasets |
| **Still hallucinates** | Fine-tuning doesn't fully solve this |
| **Outdated quickly** | New docs = retrain the model |

### RAG: The Sweet Spot

**RAG gives you the best of both worlds:**

✅ Use your private/current data  
✅ No model training required  
✅ Easy to update (just add new docs)  
✅ Grounded answers (cite sources)  
✅ Cost-effective at scale  

> **Key insight:** RAG lets you customize LLM behavior with your data without the cost and complexity of fine-tuning.

---

## The RAG Pipeline

Before diving into details, let's understand where today's topics fit in a typical RAG system:

```
┌─────────────────────────────────────────────────────────────────┐
│                     RAG PIPELINE OVERVIEW                       │
└─────────────────────────────────────────────────────────────────┘

  INGESTION (Offline - Build Time)          RETRIEVAL (Online - Query Time)
  ════════════════════════════════          ════════════════════════════════

  ┌──────────┐    ┌──────────┐    ┌──────────┐         ┌──────────┐
  │ Documents│───▶│ CHUNKING │───▶│EMBEDDINGS│───▶     │  Query   │
  └──────────┘    └──────────┘    └──────────┘         └──────────┘
                    Module 2        Module 1                 │
                                        │                    ▼
                                        ▼              ┌──────────┐
                                  ┌──────────┐         │ Embed    │
                                  │ INDEXING │         │ Query    │
                                  └──────────┘         └──────────┘
                                    Module 3                 │
                                        │                    ▼
                                        ▼              ┌──────────┐
                                  ┌──────────┐         │ Retrieve │
                                  │  Vector  │◀────────│ Top-K    │
                                  │  Store   │         └──────────┘
                                  └──────────┘               │
                                                             ▼
                                                       ┌──────────┐
                                                       │ Generate │
                                                       │ Answer   │
                                                       └──────────┘
                                                         (Day 2)
```

### Today's Focus: The Foundation

| Component | What It Does | Why It Matters |
|-----------|--------------|----------------|
| **Embeddings** | Convert text → vectors | Enables semantic search |
| **Chunking** | Split docs → smaller pieces | Right context size for retrieval |
| **Indexing** | Organize knowledge for retrieval | Determines what's findable |

> **Key insight:** Get these foundations wrong, and no amount of prompt engineering will save your RAG system.

---

## Module 1: Embeddings
**Folder:** `modules/1_embeddings/`  
**Framework:** OpenAI API (direct)

> You can't talk about embeddings without talking about their storage which is vector databases, so we will delve into details of both.

- [ ] Generate embeddings using OpenAI API
- [ ] Compute semantic similarity scores
- [ ] Visualize similarity with heatmaps
- [ ] Run `python demo.py`
- [ ] Exercises & Q&A

**Key Concepts:** Vector representations, cosine similarity, text-embedding-3-small

---

## Module 2: Chunking
**Folder:** `modules/2_chunking/`  
**Framework:** 🦜 LangChain

- [ ] Fixed-size chunking
- [ ] Recursive text splitting
- [ ] Semantic chunking
- [ ] Structure-aware splitting (Markdown/HTML)
- [ ] Build vector stores with Chroma
- [ ] Run `python demo.py`
- [ ] Exercises & Q&A

**Key Concepts:** Chunk size trade-offs, overlap strategies, token limits

---

## Module 3: Indexing Strategies
**Folder:** `modules/3_indexing/`  
**Framework:** 🦙 LlamaIndex

### ⚠️ Important Clarification: Two Types of "Indexing"

| | RAG-Level Indexing | DB-Level Indexing |
|---|---|---|
| **What** | How you organize knowledge for retrieval | How vector DBs store/search vectors |
| **Examples** | Chunking, summaries, hierarchies, metadata | HNSW, IVF, PQ, DiskANN |
| **Question** | "What knowledge is retrievable?" | "How fast can we search?" |
| **Focus** | ✅ **This module** | Covered briefly (abstracted away) |

> **This module focuses on RAG-level knowledge indexing** — the more important abstraction for building effective retrieval systems.

### Topics
- [ ] Vector Index (semantic similarity)
- [ ] Summary Index (document summaries)
- [ ] Tree Index (hierarchical retrieval)
- [ ] Keyword Table Index (traditional matching)
- [ ] Hybrid Retrieval (combining strategies)
- [ ] Run `python demo.py`
- [ ] Exercises & Q&A

**Key Concepts:** Different indexing abstractions, when to use which index
