"""CLASS 03 — MODULE 6: AGENTIC RAG — INDEPENDENT LEARNING EXAMPLES

One standalone def for every concept; no example invokes another example.
Change EXAMPLE_TO_RUN near the bottom: 1–14, 'ALL', None.

Install in activated .venv:
    pip install langchain langchain-core langchain-openai langchain-chroma langgraph python-dotenv

Set OPENAI_API_KEY for [API] examples (paid calls).
Server monitoring/log examples use SIMULATED data, never real infrastructure.
Each Chroma example builds its own in-memory sample knowledge base.
Using checkpointer=InMemorySaver() keeps history only while the process runs.
"""

# 1. DIRECT vs AGENTIC RAG [NO API]
def example_01_direct_vs_agentic_rag():
    """Fixed steps vs a decision about which available operation to use."""
    question = 'Find past database timeout incidents'
    # Fixed/direct: developer always runs the same retrieval logic.
    direct_evidence = ['T15: Database timeout']  # simulated retrieval
    print('DIRECT RAG always retrieves:', direct_evidence)
    # Conceptual agent routing: hardcoded IF illustrates choice, NOT real LLM reasoning.
    next_tool = 'search_support_tickets' if 'past' in question else 'get_server_health'
    print('SIMULATED decision (not LLM):', next_tool)


# 2. @tool DECORATOR AND .invoke() [NO API]
def example_02_custom_tool_invoke():
    """@tool wraps a Python function; invoke it with a named-argument dictionary."""
    from langchain_core.tools import tool
    @tool
    def get_server_health(server_name: str) -> str:
        """Return simulated health status for a named server."""
        return f'{server_name}: CPU utilization is 95%' if server_name == 'Server02' else f'{server_name}: CPU normal'
    print('Tool name:', get_server_health.name)
    print('Tool description:', get_server_health.description)
    print('Output:', get_server_health.invoke({'server_name': 'Server02'}))


# 3. MODEL TOOL CALL REQUEST vs EXECUTION [API]
def example_03_bind_tools_selection():
    """bind_tools lets the LLM REQUEST a tool; Python must execute separately."""
    from langchain_core.tools import tool
    from langchain_openai import ChatOpenAI
    @tool
    def get_server_health(server_name: str) -> str:
        """Get simulated CPU status of a named server."""
        return f'{server_name}: CPU 95% (simulated)'
    model = ChatOpenAI(model='gpt-4o-mini', temperature=0).bind_tools([get_server_health])
    response = model.invoke('Use the server health tool to check Server02.')
    print('LLM requested:', response.tool_calls)
    print('IMPORTANT: bind_tools/invoke here did NOT execute Python tools.')
    for call in response.tool_calls:
        if call['name'] == get_server_health.name:
            print('Manual tool execution:', get_server_health.invoke(call['args']))


# 4. create_agent AND AUTOMATIC TOOL EXECUTION [API]
def example_04_create_agent():
    """Agent runtime executes the tool requested by the model."""
    from langchain.agents import create_agent
    from langchain_core.tools import tool
    @tool
    def get_server_health(server_name: str) -> str:
        """Get simulated server CPU utilization."""
        return f'{server_name}: CPU 95% (SIMULATED)'
    agent = create_agent(model='openai:gpt-4o-mini', tools=[get_server_health])
    result = agent.invoke({'messages': [{'role': 'user', 'content': 'Check CPU of Server02 using the tool.'}]})
    print('Final answer:', result['messages'][-1].content)


# 5. MULTI-STEP TOOL SELECTION [API]
def example_05_multiple_tools():
    """Agent can request more than one tool and adjust actions based on outputs."""
    from langchain.agents import create_agent
    from langchain_core.tools import tool
    @tool
    def get_server_health(server_name: str) -> str:
        """Read simulated CPU for a server."""
        return f'{server_name}: CPU 95%'
    @tool
    def fetch_recent_logs(server_name: str) -> str:
        """Read simulated error logs for a server."""
        return f'{server_name}: Database timeouts detected'
    agent = create_agent(model='openai:gpt-4o-mini', tools=[get_server_health, fetch_recent_logs])
    result = agent.invoke({'messages': [{'role': 'user', 'content': 'Check Server02 health and logs using both tools.'}]})
    print('Answer:', result['messages'][-1].content)
    print('Tool calls:', [c['name'] for m in result['messages'] if getattr(m, 'tool_calls', None) for c in m.tool_calls])


# 6. CONVERSATION MEMORY: InMemorySaver + thread_id [API]
def example_06_in_memory_conversation():
    """Same thread_id resumes conversation; another ID isolates it."""
    from langchain.agents import create_agent
    from langgraph.checkpoint.memory import InMemorySaver
    from langchain_core.tools import tool
    @tool
    def get_server_health(server_name: str) -> str:
        """Return simulated server CPU status."""
        return f'{server_name} CPU 95% (simulated)'
    agent = create_agent(model='openai:gpt-4o-mini', tools=[get_server_health], checkpointer=InMemorySaver())
    config = {'configurable': {'thread_id': 'support_session_001'}}
    first = agent.invoke({'messages': [{'role': 'user', 'content': 'Check Server02 CPU using the tool.'}]}, config=config)
    second = agent.invoke({'messages': [{'role': 'user', 'content': 'Which server did we check?'}]}, config=config)
    print('First:', first['messages'][-1].content)
    print('Follow-up:', second['messages'][-1].content)
    print('NOTE: InMemorySaver does not survive program restart.')


# 7. TOOL CALL STRUCTURE (name, args, id) [NO API]
def example_07_tool_call_dictionary():
    """Read a simulated model tool-call dictionary with nested args."""
    tool_call = {'name': 'fetch_recent_logs', 'args': {'server_name': 'Server02'}, 'id': 'call_example'}
    print('Tool:', tool_call['name'])
    print('All arguments:', tool_call['args'])
    print('Server name:', tool_call['args']['server_name'])
    print('ID:', tool_call['id'])


# 8. TRACE AGENT MESSAGES AND TOOL CALLS [API]
def example_08_agent_trace():
    """Inspect HumanMessage / AIMessage / ToolMessage and requested tool names."""
    from langchain.agents import create_agent
    from langchain_core.tools import tool
    @tool
    def fetch_recent_logs(server_name: str) -> str:
        """Return simulated recent server logs."""
        return f'{server_name}: database timeout (simulated)'
    agent = create_agent(model='openai:gpt-4o-mini', tools=[fetch_recent_logs])
    result = agent.invoke({'messages': [{'role': 'user', 'content': 'Use the logs tool for Server02.'}]})
    for message in result['messages']:
        print(type(message).__name__, ':', message.content)
        for call in getattr(message, 'tool_calls', []) or []:
            print('  Requested:', call['name'], 'args=', call['args'])


# 9. ERROR HANDLING IN A TOOL [NO API]
def example_09_tool_try_except():
    """Handle expected failures; avoid confusing failed retrieval with zero matches."""
    from langchain_core.tools import tool
    @tool
    def search_support_tickets(query: str) -> str:
        """Simulate search success, no matches, or unavailable storage."""
        try:
            if query == 'offline':
                raise ConnectionError('Simulated ChromaDB unavailable')
            docs = ['T15: database timeout'] if 'database' in query else []
            if not docs:
                return 'No matching tickets found.'
            return '\n'.join(docs)
        except ConnectionError:
            return 'Ticket search unavailable; please try again later.'
    for q in ['database timeout', 'printer issue', 'offline']:
        print(q, '->', search_support_tickets.invoke({'query': q}))


# 10. CHROMA SEARCH TOOL + DOCUMENT METADATA [API]
def example_10_chroma_retrieval_tool():
    """Independent Chroma store, tickets, retriever tool, and test invocation."""
    from langchain_core.documents import Document
    from langchain_core.tools import tool
    from langchain_openai import OpenAIEmbeddings
    from langchain_chroma import Chroma
    store = Chroma(collection_name='agent_10_demo', embedding_function=OpenAIEmbeddings(model='text-embedding-3-small'))
    store.add_documents([
        Document(page_content='Database connection timeout. Check connection pool.', metadata={'ticket_id': 'T15'}),
        Document(page_content='Users unable to sign in after password reset.', metadata={'ticket_id': 'T1'}),
    ])
    @tool
    def search_support_tickets(query: str) -> str:
        """Search sample historical tickets and return ID plus ticket content."""
        try:
            docs = store.similarity_search(query, k=2)
        except Exception:
            return 'Ticket search failed; please retry later.'
        results = [f"Ticket: {d.metadata.get('ticket_id', 'Unknown')}\nContent: {d.page_content}" for d in docs]
        return '\n\n'.join(results) if results else 'No tickets found.'
    print(search_support_tickets.invoke({'query': 'database timeouts'}))


# 11. COMPLETE AGENTIC RAG: 3 TOOLS + CHROMA [API]
def example_11_complete_agentic_rag():
    """End-to-end agent, independent sample Chroma collection and three tools."""
    from langchain.agents import create_agent
    from langchain_core.tools import tool
    from langchain_core.documents import Document
    from langchain_openai import OpenAIEmbeddings
    from langchain_chroma import Chroma
    store = Chroma(collection_name='agent_11_demo', embedding_function=OpenAIEmbeddings(model='text-embedding-3-small'))
    store.add_documents([
        Document(page_content='Database connection pool exhausted; review pool limits.', metadata={'ticket_id': 'T15'}),
        Document(page_content='Database timeout: investigate slow queries.', metadata={'ticket_id': 'T21'}),
        Document(page_content='Password reset error: verify the new password.', metadata={'ticket_id': 'T1'}),
    ])
    @tool
    def get_server_health(server_name: str) -> str:
        """Check simulated CPU load for a named server."""
        return f'{server_name}: CPU 95% (simulated)' if server_name == 'Server02' else f'{server_name}: CPU normal (simulated)'
    @tool
    def fetch_recent_logs(server_name: str) -> str:
        """Retrieve simulated application logs for a named server."""
        return f'{server_name}: database timeout errors (simulated)'
    @tool
    def search_support_tickets(query: str) -> str:
        """Search historical support-ticket knowledge base for similar incidents."""
        try:
            docs = store.similarity_search(query, k=3)
        except Exception:
            return 'Historical ticket search unavailable.'
        results = [f"[{d.metadata.get('ticket_id', 'Unknown')}] {d.page_content}" for d in docs]
        return '\n'.join(results) if results else 'No matching tickets.'
    agent = create_agent(model='openai:gpt-4o-mini',
                         tools=[get_server_health, fetch_recent_logs, search_support_tickets],
                         system_prompt='Use tools for server facts. Distinguish simulated observations from historical evidence. Do not claim a confirmed root cause without proof.')
    result = agent.invoke({'messages': [{'role': 'user', 'content': 'Use the tools to investigate Server02 CPU, inspect its logs, and find similar historical tickets.'}]})
    for message in result['messages']:
        if getattr(message, 'tool_calls', None):
            print('REQUESTED:', [(c['name'], c['args']) for c in message.tool_calls])
    print('FINAL ANSWER:\n', result['messages'][-1].content)


# 12. .join() AND EMPTY LIST FALLBACK [NO API]
def example_12_join_and_fallback():
    """Review Python used to format the Chroma result into one string."""
    results = ['Ticket T15: Database timeout', 'Ticket T21: Slow SQL query']
    print('With matches:\n' + ('\n\n'.join(results) if results else 'No tickets found.'))
    results = []
    print('Without matches:', '\n\n'.join(results) if results else 'No tickets found.')


# 13. READ AGENT FINAL ANSWER WITH [-1] [NO API]
def example_13_result_messages_negative_index():
    """A small simulation to illustrate Python negative indexing, no LLM used."""
    messages = ['HumanMessage', 'AIMessage(tool call)', 'ToolMessage', 'AIMessage(final answer)']
    print('All messages:', messages)
    print('Last message:', messages[-1])
    print('Second-to-last:', messages[-2])
    print('In a completed agent run: result["messages"][-1].content')


# 14. MULTI-STEP MEMORY + CHROMA ALL IN ONE [API]
def example_14_agentic_rag_with_memory():
    """Independent minimal three-tool Agentic RAG + InMemorySaver example."""
    from langchain.agents import create_agent
    from langgraph.checkpoint.memory import InMemorySaver
    from langchain_core.tools import tool
    from langchain_core.documents import Document
    from langchain_openai import OpenAIEmbeddings
    from langchain_chroma import Chroma
    store = Chroma(collection_name='agent_14_demo', embedding_function=OpenAIEmbeddings(model='text-embedding-3-small'))
    store.add_documents([Document(page_content='T15: Database timeout caused by exhausted connection pool.',
                                  metadata={'ticket_id': 'T15'})])
    @tool
    def get_server_health(server_name: str) -> str:
        """Return simulated current server CPU utilization."""
        return f'{server_name}: CPU 95% (simulated)'
    @tool
    def fetch_recent_logs(server_name: str) -> str:
        """Return simulated recent application errors."""
        return f'{server_name}: database timeout (simulated)'
    @tool
    def search_support_tickets(query: str) -> str:
        """Search previous support tickets in ChromaDB."""
        try:
            docs = store.similarity_search(query, k=1)
        except Exception:
            return 'Ticket retrieval unavailable.'
        return '\n'.join(f"{d.metadata.get('ticket_id', '?')}: {d.page_content}" for d in docs) if docs else 'No tickets.'
    agent = create_agent(model='openai:gpt-4o-mini',
                         tools=[get_server_health, fetch_recent_logs, search_support_tickets],
                         checkpointer=InMemorySaver())
    config = {'configurable': {'thread_id': 'support_session_001'}}
    first = agent.invoke({'messages': [{'role': 'user', 'content': 'Use health and logs tools to investigate Server02.'}]}, config=config)
    second = agent.invoke({'messages': [{'role': 'user', 'content': 'Use historical ticket search for similar issues.'}]}, config=config)
    print('First response:', first['messages'][-1].content)
    print('Follow-up:', second['messages'][-1].content)
    print('InMemorySaver is NOT durable across Python restarts.')


EXAMPLE_TO_RUN = 1  # 1–14, 'ALL', or None; ALL makes multiple paid API calls!


def main():
    """Launcher only. Examples are completely independent of one another."""
    examples = {
        1: example_01_direct_vs_agentic_rag, 2: example_02_custom_tool_invoke,
        3: example_03_bind_tools_selection, 4: example_04_create_agent,
        5: example_05_multiple_tools, 6: example_06_in_memory_conversation,
        7: example_07_tool_call_dictionary, 8: example_08_agent_trace,
        9: example_09_tool_try_except, 10: example_10_chroma_retrieval_tool,
        11: example_11_complete_agentic_rag, 12: example_12_join_and_fallback,
        13: example_13_result_messages_negative_index, 14: example_14_agentic_rag_with_memory,
    }
    if EXAMPLE_TO_RUN is None:
        print('Set EXAMPLE_TO_RUN to 1–14 or ALL.')
        return
    selected = examples.items() if str(EXAMPLE_TO_RUN).upper() == 'ALL' else [(EXAMPLE_TO_RUN, examples[EXAMPLE_TO_RUN])]
    for number, example in selected:
        print(f'\n{"=" * 60}\nEXAMPLE {number}: {example.__name__}\n{"=" * 60}')
        example()


if __name__ == '__main__':
    main()
