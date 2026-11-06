"""CLASS 03 — MODULE 4: RAG PIPELINE — INDEPENDENT LEARNING EXAMPLES

Each example_XX() is self-contained: it does NOT call any other example_XX().
Change EXAMPLE_TO_RUN at the bottom to a number, 'ALL', or None.
Default example 1 is free and needs only langchain-core.

Install (in your course venv):
    pip install langchain-core langchain-openai langchain-chroma langchain-text-splitters python-dotenv

Set OPENAI_API_KEY for examples marked [API]. They make paid API requests.
Examples using Chroma create an independent, in-memory collection on each run.
Sample tickets are intentionally tiny; real retrieval scores/wording can vary.
"""

# 1. RAG = RETRIEVE -> AUGMENT -> GENERATE [NO API]
def example_01_rag_three_stages():
    """Learn the three phases without hiding them behind a framework."""
    question = "How do I fix a password reset login error?"
    # RETRIEVE: simulated result from a knowledge base, not a real DB search.
    retrieved_ticket = "Ticket T1: After reset, use the new password to sign in."
    # AUGMENT: add retrieved evidence to a prompt.
    prompt = f"Answer using only this ticket:\n{retrieved_ticket}\nQuestion: {question}"
    # GENERATE: simulated output; a real app sends 'prompt' to an LLM.
    simulated_answer = "Try signing in with the newly reset password (T1)."
    print('QUESTION:', question)
    print('RETRIEVED:', retrieved_ticket)
    print('AUGMENTED PROMPT:\n', prompt)
    print('SIMULATED ANSWER:', simulated_answer)


# 2. LANGCHAIN DOCUMENT AND METADATA [NO API]
def example_02_document_page_content_metadata():
    """Document.page_content stores text; Document.metadata stores labels."""
    from langchain_core.documents import Document
    doc = Document(page_content="Database connection timed out.",
                   metadata={"ticket_id": "T15", "category": "Database"})
    print('Text:', doc.page_content)
    print('Metadata:', doc.metadata)
    print('Ticket ID:', doc.metadata.get('ticket_id', 'Unknown'))


# 3. CHROMA DOCUMENT STORAGE + SIMILARITY SEARCH [API]
def example_03_chroma_similarity_search():
    """Create and search a self-contained in-memory collection."""
    from langchain_core.documents import Document
    from langchain_openai import OpenAIEmbeddings
    from langchain_chroma import Chroma
    docs = [
        Document(page_content="Password reset causes login failure.", metadata={"ticket_id": "T1"}),
        Document(page_content="Database connection timeout error.", metadata={"ticket_id": "T2"}),
        Document(page_content="MFA token expired during sign in.", metadata={"ticket_id": "T3"}),
    ]
    store = Chroma(collection_name="rag_03_demo", embedding_function=OpenAIEmbeddings(model="text-embedding-3-small"))
    store.add_documents(docs)
    matches = store.similarity_search("Cannot log in after password reset", k=2)
    for doc in matches:
        print(doc.metadata.get('ticket_id'), doc.page_content)


# 4. RETRIEVER: SEARCH INTERFACE [API]
def example_04_retriever_invoke():
    """as_retriever().invoke(question) returns Documents, not an LLM answer."""
    from langchain_core.documents import Document
    from langchain_openai import OpenAIEmbeddings
    from langchain_chroma import Chroma
    store = Chroma(collection_name="rag_04_demo", embedding_function=OpenAIEmbeddings(model="text-embedding-3-small"))
    store.add_documents([Document(page_content="Login failure after password reset."),
                         Document(page_content="Database timeout occurred.")])
    retriever = store.as_retriever(search_kwargs={"k": 1})
    matches = retriever.invoke("password login problem")
    for doc in matches:
        print('Retrieved:', doc.page_content)


# 5. PROMPT TEMPLATE AND GROUNDEDNESS [NO API]
def example_05_prompt_template():
    """PromptTemplate fills {context} and {question} placeholders."""
    from langchain_core.prompts import PromptTemplate
    template = PromptTemplate.from_template(
        "Use ONLY the context. If missing, say 'I don't know'.\n"
        "Context:\n{context}\nQuestion: {question}\nAnswer:"
    )
    prompt = template.invoke({"context": "Ticket T1: Reset your password and retry.",
                              "question": "How can I log in?"})
    print(prompt.to_string())


# 6. LLM + StrOutputParser [API]
def example_06_llm_str_output_parser():
    """The LLM returns a message; StrOutputParser extracts a string."""
    from langchain_openai import ChatOpenAI
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    prompt = ChatPromptTemplate.from_template('Explain {term} in one simple sentence.')
    chain = prompt | ChatOpenAI(model="gpt-4o-mini", temperature=0) | StrOutputParser()
    print(chain.invoke({"term": "RAG"}))


# 7. LCEL PIPE + RunnableLambda [NO API]
def example_07_lcel_pipe():
    """The | operator passes each runnable's output to the next one."""
    from langchain_core.runnables import RunnableLambda
    def make_uppercase(text):
        return text.upper()
    def add_exclamation(text):
        return text + '!'
    chain = RunnableLambda(make_uppercase) | RunnableLambda(add_exclamation)
    print(chain.invoke('hello'))  # HELLO!


# 8. RunnablePassthrough + PARALLEL INPUTS [NO API]
def example_08_runnable_passthrough():
    """Retain the original question while computing another field."""
    from langchain_core.runnables import RunnableLambda, RunnablePassthrough
    chain = {"question": RunnablePassthrough(),
             "context": RunnableLambda(lambda q: f"Simulated evidence for: {q}")}
    print(chain.invoke('database timeout'))


# 9. COMPLETE BASIC RAG CHAIN [API]
def example_09_complete_rag_chain():
    """Retriever -> formatted context -> prompt -> LLM -> string."""
    from langchain_core.documents import Document
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnableLambda, RunnablePassthrough
    from langchain_openai import OpenAIEmbeddings, ChatOpenAI
    from langchain_chroma import Chroma
    store = Chroma(collection_name="rag_09_demo", embedding_function=OpenAIEmbeddings(model="text-embedding-3-small"))
    store.add_documents([
        Document(page_content="After password reset, use your new password.", metadata={"ticket_id": "T1"}),
        Document(page_content="A locked account requires an administrator unlock.", metadata={"ticket_id": "T2"}),
    ])
    retriever = store.as_retriever(search_kwargs={"k": 2})
    def format_docs(docs):
        return '\n'.join(f"[{d.metadata.get('ticket_id', '?')}] {d.page_content}" for d in docs)
    prompt = ChatPromptTemplate.from_template(
        "Answer only from the tickets; cite ticket IDs. If unknown, say so.\n{context}\nQuestion: {question}")
    chain = ({"context": retriever | RunnableLambda(format_docs), "question": RunnablePassthrough()}
             | prompt | ChatOpenAI(model="gpt-4o-mini", temperature=0) | StrOutputParser())
    print(chain.invoke('How should I sign in after resetting my password?'))


# 10. INLINE CITATIONS FROM DOCUMENT METADATA [NO API]
def example_10_inline_citations():
    """Format evidence with ticket IDs; citations must refer to real retrieved items."""
    from langchain_core.documents import Document
    docs = [Document(page_content="Reset the password.", metadata={"ticket_id": "T1"}),
            Document(page_content="Check MFA enrollment.", metadata={"ticket_id": "T3"})]
    context = '\n'.join(f"[{d.metadata['ticket_id']}] {d.page_content}" for d in docs)
    print('Evidence provided to LLM:\n' + context)
    print('Illustrative grounded answer: Reset the password [T1] and check MFA [T3].')


# 11. FALLBACK WHEN RETRIEVAL IS EMPTY [NO API]
def example_11_empty_retrieval_fallback():
    """Do not invent an answer when no supporting documents were found."""
    retrieved_docs = []  # Simulated no results
    if not retrieved_docs:
        answer = "I don't have enough information in the knowledge base."
    else:
        answer = '\n'.join(retrieved_docs)
    print(answer)


# 12. STUFF vs MAP-REDUCE vs REFINE [NO API]
def example_12_stuff_map_reduce_refine():
    """Explain three strategies with visible Python operations (no real LLM)."""
    docs = ['T1: Login failure.', 'T2: Password expired.', 'T3: MFA token invalid.']
    stuffed_context = '\n'.join(docs)
    print('STUFF: all text in one context:\n', stuffed_context)
    mapped_summaries = [f"Summary of {d}" for d in docs]  # Simulates LLM map calls
    print('MAP: independent simulated summaries:', mapped_summaries)
    print('REDUCE: simulated combined summary:', '; '.join(mapped_summaries))
    running_summary = ''
    for doc in docs:
        running_summary = f"{running_summary} {doc}".strip()  # Conceptual refine; no LLM
    print('REFINE: update a running result per document:', running_summary)


# 13. METADATA FILTERING [API]
def example_13_metadata_filtering():
    """Limit retrieval to a category before searching for similar content."""
    from langchain_core.documents import Document
    from langchain_openai import OpenAIEmbeddings
    from langchain_chroma import Chroma
    store = Chroma(collection_name="rag_13_demo", embedding_function=OpenAIEmbeddings(model="text-embedding-3-small"))
    store.add_documents([
        Document(page_content="Login failed after reset.", metadata={"category": "Authentication"}),
        Document(page_content="Database timed out.", metadata={"category": "Database"}),
    ])
    docs = store.similarity_search('login issue', k=2, filter={"category": "Authentication"})
    for d in docs:
        print(d.metadata, d.page_content)


# 14. STREAMING LLM OUTPUT [API]
def example_14_streaming():
    """.stream() yields chunks as the LLM generates text."""
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    chain = (ChatPromptTemplate.from_template('Define {term} briefly.')
             | ChatOpenAI(model='gpt-4o-mini') | StrOutputParser())
    for chunk in chain.stream({"term": "retriever"}):
        print(chunk, end='', flush=True)
    print()


# 15. MULTI-TURN RAG BY EXPLICIT HISTORY [API]
def example_15_multi_turn_history():
    """Pass history yourself; this is distinct from checkpointer-based agent memory."""
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain_core.messages import HumanMessage, AIMessage
    from langchain_core.output_parsers import StrOutputParser
    prompt = ChatPromptTemplate.from_messages([
        ('system', 'Use the supplied context; do not invent facts. Context: {context}'),
        MessagesPlaceholder('history'),
        ('human', '{question}'),
    ])
    history = [HumanMessage(content='Tell me about ticket T15.'),
               AIMessage(content='Ticket T15 concerns database timeouts.')]
    response = (prompt | ChatOpenAI(model='gpt-4o-mini', temperature=0) | StrOutputParser()).invoke({
        'context': 'T15: database timeouts. Resolution: review connection pool.',
        'history': history,
        'question': 'What was its proposed resolution?',
    })
    print(response)


EXAMPLE_TO_RUN = 1  # 1–15, 'ALL', or None; ALL makes many paid API calls!


def main():
    """Only the launcher dispatches examples; examples never depend on each other."""
    examples = {
        1: example_01_rag_three_stages, 2: example_02_document_page_content_metadata,
        3: example_03_chroma_similarity_search, 4: example_04_retriever_invoke,
        5: example_05_prompt_template, 6: example_06_llm_str_output_parser,
        7: example_07_lcel_pipe, 8: example_08_runnable_passthrough,
        9: example_09_complete_rag_chain, 10: example_10_inline_citations,
        11: example_11_empty_retrieval_fallback, 12: example_12_stuff_map_reduce_refine,
        13: example_13_metadata_filtering, 14: example_14_streaming,
        15: example_15_multi_turn_history,
    }
    if EXAMPLE_TO_RUN is None:
        print('Set EXAMPLE_TO_RUN to a number or ALL.')
        return
    selected = examples.items() if str(EXAMPLE_TO_RUN).upper() == 'ALL' else [(EXAMPLE_TO_RUN, examples[EXAMPLE_TO_RUN])]
    for number, example in selected:
        print(f'\n{"=" * 60}\nEXAMPLE {number}: {example.__name__}\n{"=" * 60}')
        example()


if __name__ == '__main__':
    main()
