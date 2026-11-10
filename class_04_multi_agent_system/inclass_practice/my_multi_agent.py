"""Interview Kickstart / Class 04 — Multi-Agent Systems reference.

18 self-contained teaching functions, selected by EXAMPLE_TO_RUN at the bottom.
These are illustrative learning implementations, not verbatim instructor code.

Set up in your VS Code PowerShell virtual environment:
    python -m pip install -U langgraph langchain langchain-openai

Examples 1-4 and 6 (and the conceptual parts of 2-3) need only Python.
Examples 5, 7-8, 11-14, 16-18 require LangGraph.
Examples 9-10 and 15 require an OPENAI_API_KEY and invoke an LLM (costs may apply).
Example 9 uses a supervisor/subagent-as-tool; Example 15 embeds agents inside
LangGraph workers. These are TWO DIFFERENT orchestration patterns.

IMPORTANT: Sample hotel/pricing information is fictitious for learning.
"""
from __future__ import annotations

from typing import TypedDict, Annotated
import operator


# ---------------------------------------------------------------------------
# 01. MULTI-AGENT SYSTEMS: recognize specialist responsibilities.
# INPUT: None. OUTPUT: Print example agent roles.
# ---------------------------------------------------------------------------
def example_01_multi_agent_basics():
    specialists = {
        "search": "Find hotels and flights",
        "planner": "Create a day-by-day itinerary",
    }
    for name, responsibility in specialists.items():
        print(f"{name}: {responsibility}")
    print("A multi-agent system coordinates specialists to solve a larger task.")


# ---------------------------------------------------------------------------
# 02. WHEN TO USE MULTIPLE AGENTS.
# INPUT: Scope of request. OUTPUT: Simple recommendation.
# ---------------------------------------------------------------------------
def example_02_when_multi_agent():
    tasks = ["Find hotels", "Build itinerary", "Research transport"]
    if len(tasks) == 1:
        print("Start with one agent or even one tool.")
    else:
        print("Specialized agents may help, if complexity justifies overhead.")
    print("Task candidates:", tasks)


# ---------------------------------------------------------------------------
# 03. SEQUENTIAL vs PARALLEL vs ORCHESTRATOR.
# INPUT: None. OUTPUT: Illustrative task ordering.
# ---------------------------------------------------------------------------
def example_03_workflow_patterns():
    tasks = ["hotel research", "itinerary research"]
    print("Sequential: search -> itinerary")
    print("Parallel: search and itinerary are eligible to run together")
    print("Orchestrator: creates assignments:", tasks)
    print("Orchestrator = WHO decides; parallel = HOW tasks execute")


# ---------------------------------------------------------------------------
# 04. FRAMEWORK ROLES: Python, LangChain, LangGraph.
# INPUT: None. OUTPUT: Concept map.
# ---------------------------------------------------------------------------
def example_04_frameworks():
    roles = {
        "Python": "functions and business rules",
        "LangChain": "LLM, tools, and ready-made agents",
        "LangGraph": "state, edges, routing, parallel tasks, persistence",
    }
    for framework, role in roles.items():
        print(f"{framework}: {role}")


# ---------------------------------------------------------------------------
# 05. StateGraph, state, node, and edges.
# INPUT: Destination. OUTPUT: Updated graph state.
# ---------------------------------------------------------------------------
def example_05_stategraph_nodes_edges():
    from langgraph.graph import StateGraph, START, END

    class State(TypedDict):
        destination: str
        answer: str

    def planner_node(state: State) -> dict:
        return {"answer": f"Plan a trip to {state['destination']}"}

    builder = StateGraph(State)
    builder.add_node("planner", planner_node)  # Registration; does not run it
    builder.add_edge(START, "planner")           # Fixed execution path
    builder.add_edge("planner", END)
    graph = builder.compile()                   # Compile; does not run nodes
    result = graph.invoke({"destination": "Paris", "answer": ""})
    print(result)


# ---------------------------------------------------------------------------
# 06. TypedDict / Annotated / reducer metadata WITHOUT LangGraph.
# INPUT: Two lists. OUTPUT: Combined lists.
# ---------------------------------------------------------------------------
def example_06_typeddict_annotated_reducer():
    from typing import get_type_hints

    class State(TypedDict):
        agent_results: Annotated[list[dict], operator.add]

    annotation = get_type_hints(State, include_extras=True)["agent_results"]
    reducer = annotation.__metadata__[0]  # Reads, doesn't automatically run
    old = [{"agent": "search"}]
    update = [{"agent": "planner"}]
    print("Reducer:", reducer)
    print("Combined:", reducer(old, update))


# ---------------------------------------------------------------------------
# 07. Conditional edges / routing: router is not a graph node.
# INPUT: User query. OUTPUT: Selected specialist's answer.
# ---------------------------------------------------------------------------
from typing import TypedDict
from langgraph.graph import StateGraph, START, END


# Define at module level, outside the function.
class ConditionalEdgesState(TypedDict):
    user_query: str
    answer: str


def example_07_conditional_edges():

    def route(state: ConditionalEdgesState) -> str:
        return (
            "search"
            if "hotel" in state["user_query"].lower()
            else "planner"
        )

    def search(state: ConditionalEdgesState) -> dict:
        return {"answer": "Simulated hotel result"}

    def planner(state: ConditionalEdgesState) -> dict:
        return {"answer": "Simulated itinerary"}

    b = StateGraph(ConditionalEdgesState)

    b.add_node("search_node", search)
    b.add_node("planner_node", planner)

    b.add_conditional_edges(
        START,
        route,
        {
            "search": "search_node",
            "planner": "planner_node",
        },
    )

    b.add_edge("search_node", END)
    b.add_edge("planner_node", END)

    graph = b.compile()

    result = graph.invoke({
        "user_query": "Find hotels",
        "answer": "",
    })

    print(result)


# ---------------------------------------------------------------------------
# 08. Checkpointers: same thread_id restores previous state.
# INPUT: Two requests; OUTPUT: Second request still knows Paris.
# ---------------------------------------------------------------------------
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver


# Outside the function so LangGraph can resolve this type.
class CheckpointerState(TypedDict):
    destination: str
    latest_request: str
    answer: str


def example_08_checkpointer():

    def node(state: CheckpointerState) -> dict:
        return {
            "answer": (
                f"{state['destination']}: "
                f"{state['latest_request']}"
            )
        }

    b = StateGraph(CheckpointerState)

    b.add_node("node", node)
    b.add_edge(START, "node")
    b.add_edge("node", END)

    graph = b.compile(checkpointer=InMemorySaver())

    config = {  "configurable": {"thread_id": "travel_101"}   }
    config2 = {  "configurable": {"thread_id": "travel_102"}   }
    # First call: initialize this thread.
    result = graph.invoke(
        {
            "destination": "Paris",
            "latest_request": "Plan trip",
            "answer": "",
        },
        config=config,
    )
    print(result["answer"])

    # Same thread: destination is remembered.
    result = graph.invoke(
        { "destination": "India", "latest_request": "Add Taj"},
        config=config,
    )
    print(result["answer"])

    print(
        "InMemorySaver is not durable across "
        "a Python process restart."
    )
# ---------------------------------------------------------------------------
# 09. SUBAGENT AS TOOL: Supervisor -> delegation tool -> Search Agent -> tool.
# INPUT: User request. OUTPUT: Supervisor reply. REQUIRES OPENAI_API_KEY.
# ---------------------------------------------------------------------------
def example_09_supervisor_and_subagent():
    from langchain_openai import ChatOpenAI
    from langchain.agents import create_agent
    from langchain_core.tools import tool

    @tool
    def find_hotels(city: str) -> str:
        """Return fictitious hotel information for a city."""
        print(f"Inside find hotel tool {city}")
        return f"Sample hotel in {city}; this is simulated, not a live search."

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    hotel_search_agent = create_agent(
        model=llm, tools=[find_hotels],
        system_prompt="You research hotels. Use find_hotels. Clearly label simulated data."
    )

    @tool
    def search_travel(request: str) -> str:
        """Delegate a hotel question to a specialized Hotel Search Agent."""
        print(f"Inside search travel tool : {request}")
        result = hotel_search_agent.invoke({"messages": [{"role": "user", "content": request}]})
        re = str(result["messages"][-1].content)
        print(f"re************ {re} **************")
        return re

    supervisor = create_agent(
        model=llm, tools=[search_travel],
        system_prompt="You are a supervisor. Delegate hotel requests to search_travel."
    )
    result = supervisor.invoke({"messages": [{"role": "user", "content": "Find hotels in Paris"}]})
    print(f"@@@@@@@@@@@@@@@@@@@@@ final result ---------- {result["messages"][-1].content} ^^^^^^^^^^^")


# ---------------------------------------------------------------------------
# 10. TOOL CALL LOOP + AGENT STREAMING.
# The hidden internal model->tool->model loop is NOT the Python for loop below.
# INPUT: Question. OUTPUT: Agent updates. REQUIRES OPENAI_API_KEY.
# ---------------------------------------------------------------------------
def example_10_agent_tool_loop_streaming():
    from langchain_openai import ChatOpenAI
    from langchain.agents import create_agent
    from langchain_core.tools import tool

    @tool
    def find_hotels(city: str) -> str:
        """Return fictitious hotel options for a city."""
        print("ACTUAL TOOL EXECUTION:", city)
        return f"Sample Hotel in {city} (fictitious) Hotel Lumiere: $190/night.No hotels within budget found in the city center."

    agent = create_agent(
        model=ChatOpenAI(model="gpt-4o-mini", temperature=0),
        tools=[find_hotels],
        system_prompt="For hotel requests use find_hotels; label all output simulated."
    )
    input_data = {"messages": [{"role": "user", "content": "Find a hotel in Paris under $150 per night.if not found then explore near by city. even near by city does not have anything below 150 then get me the least hotel rate found"}]}
    #for update in agent.stream(input_data, stream_mode="updates"):
    #    print("AGENT UPDATE:", update)
    result = agent.invoke(input_data)
    print(f"@@@@@@@@@@@@@@@@@@@@@ final result ---------- {result["messages"][-1].content} ^^^^^^^^^^^")


# ---------------------------------------------------------------------------
# 11. ORCHESTRATOR-WORKER SEQUENTIAL workflow (fixed edges).
# INPUT: Graph input. OUTPUT: Results collected from two fixed workers.
# ---------------------------------------------------------------------------
def example_11_orchestrator_worker_sequential():
    from langgraph.graph import StateGraph, START, END

    class State(TypedDict):
        tasks: list[str]
        agent_results: Annotated[list[str], operator.add]
        final_answer: str

    def orchestrator(state: State):
        return {"tasks": ["hotels", "itinerary"]}

    def search(state: State):
        return {"agent_results": ["Hotel: simulated example"]}

    def planner(state: State):
        return {"agent_results": ["Itinerary: simulated example"]}

    def synthesis(state: State):
        return {"final_answer": "\n".join(state["agent_results"])}

    b = StateGraph(State)
    for name, fn in [("orchestrator", orchestrator), ("search", search), ("planner", planner), ("synthesis", synthesis)]:
        b.add_node(name, fn)
    b.add_edge(START, "orchestrator")
    b.add_edge("orchestrator", "search")
    b.add_edge("search", "planner")
    b.add_edge("planner", "synthesis")
    b.add_edge("synthesis", END)
    result = b.compile().invoke({"tasks": [], "agent_results": [], "final_answer": ""})
    print(result["final_answer"])


# ---------------------------------------------------------------------------
# 12. DYNAMIC PARALLELIZATION via Send.
# INPUT: Task list from orchestrator. OUTPUT: Two worker results.
# ---------------------------------------------------------------------------
class example_12_parallel_sendState(TypedDict):
        tasks: list[str]
        agent_results: Annotated[list[str], operator.add]

def example_12_parallel_send():
    from langgraph.graph import StateGraph, START, END
    from langgraph.types import Send

    

    class WorkerState(TypedDict):
        task: str

    def orchestrator(state: example_12_parallel_sendState):
        return {"tasks": ["hotels", "itinerary"]}

    def route(state: example_12_parallel_sendState):
        # Creates TWO invocations of the SAME registered worker node.
        return [Send("worker", {"task": task}) for task in state["tasks"]]

    def worker(state: WorkerState):
        print("Worker input:", state)
        # WorkerState is input; returned dict updates SHARED State.
        return {"agent_results": [f"Processed {state['task']}"]}

    b = StateGraph(example_12_parallel_sendState)
    b.add_node("orchestrator", orchestrator)
    b.add_node("worker", worker)
    b.add_edge(START, "orchestrator")
    b.add_conditional_edges("orchestrator", route, ["worker"])
    b.add_edge("worker", END)
    result = b.compile().invoke({"tasks": [], "agent_results": []})
    print("Combined results (order need not be assumed):", result["agent_results"])


# ---------------------------------------------------------------------------
# 13. SYNTHESIS: reducer collects, synthesizer formats.
# INPUT: Two Send worker outputs. OUTPUT: Structured final answer.
# ---------------------------------------------------------------------------
class example_13_State(TypedDict):
        tasks: list[str]
        agent_results: Annotated[list[dict], operator.add]
        final_answer: str
def example_13_synthesizing_results():
    from langgraph.graph import StateGraph, START, END
    from langgraph.types import Send

    

    class WorkerState(TypedDict):
        task: str

    def orchestrator(state: example_13_State):
        return {"tasks": ["search", "planner"]}

    def dispatch(state: example_13_State):
        return [Send("worker", {"task": t}) for t in state["tasks"]]

    def worker(state: WorkerState):
        name = state["task"]
        answer = "Simulated hotel" if name == "search" else "Simulated itinerary"
        return {"agent_results": [{"agent": name, "answer": answer}]}

    def synthesizer(state: example_13_State):
        by_agent = {item["agent"]: item["answer"] for item in state["agent_results"]}
        return {"final_answer": f"Hotel: {by_agent.get('search')}\nPlan: {by_agent.get('planner')}"}

    b = StateGraph(example_13_State)
    for name, fn in [("orchestrator", orchestrator), ("worker", worker), ("synthesizer", synthesizer)]:
        b.add_node(name, fn)
    b.add_edge(START, "orchestrator")
    b.add_conditional_edges("orchestrator", dispatch, ["worker"])
    b.add_edge("worker", "synthesizer")
    b.add_edge("synthesizer", END)
    result = b.compile().invoke({"tasks": [], "agent_results": [], "final_answer": ""})
    print(result["final_answer"])


# ---------------------------------------------------------------------------
# 14. FULL IK-STYLE TravelPlannerState WITH simulated specialists.
# INPUT: User query. OUTPUT: State fields, messages, and final answer.
# ---------------------------------------------------------------------------
import operator
from typing import Annotated, TypedDict
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage

class AgentTask(TypedDict):
    agent: str
    description: str

class example_14_State(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    user_query: str
    tasks: list[AgentTask]
    requires_synthesis: bool
    agent_results: Annotated[list[dict], operator.add]
    final_answer: str
    search_messages: Annotated[list[AnyMessage], operator.add]
    planner_messages: Annotated[list[AnyMessage], operator.add]

def example_14_complete_travelplanner_state():
    from langchain_core.messages import AnyMessage, HumanMessage, AIMessage
    from langgraph.graph import StateGraph, START, END
    from langgraph.types import Send

    

    
    class WorkerState(TypedDict):
        task: AgentTask

    def orchestrator(state: example_14_State):
        return {
            "tasks": [
                {"agent": "search", "description": "Find hotels in Paris"},
                {"agent": "planner", "description": "Create three-day Paris itinerary"},
            ],
            "requires_synthesis": True,
        }

    def assign_workers(state: example_14_State):
        return [Send("worker", {"task": t}) for t in state["tasks"]]

    def worker(state: WorkerState):
        task = state["task"]
        name = task["agent"]
        answer = ("Sample hotel: Central Hotel (fictional)" if name == "search"
                  else "Sample itinerary: Eiffel Tower, Louvre, Montmartre")
        # The return value is an update to TravelPlannerState, NOT WorkerState.
        update = {"agent_results": [{"agent": name, "result": answer}]}
        if name == "search":
            update["search_messages"] = [AIMessage(content=answer)]
        else:
            update["planner_messages"] = [AIMessage(content=answer)]
        return update

    def synthesizer(state: example_14_State):
        results = {r["agent"]: r["result"] for r in state["agent_results"]}
        answer = f"Hotel: {results['search']}\nItinerary: {results['planner']}"
        return {"final_answer": answer, "messages": [AIMessage(content=answer)]}

    b = StateGraph(example_14_State)
    for name, fn in [("orchestrator", orchestrator), ("worker", worker), ("synthesizer", synthesizer)]:
        b.add_node(name, fn)
    b.add_edge(START, "orchestrator")
    b.add_conditional_edges("orchestrator", assign_workers, ["worker"])
    b.add_edge("worker", "synthesizer")
    b.add_edge("synthesizer", END)
    graph = b.compile()
    result = graph.invoke({
        "messages": [HumanMessage(content="Find hotels and plan a Paris trip")],
        "user_query": "Find hotels and plan a Paris trip",
        "tasks": [], "requires_synthesis": False,
        "agent_results": [], "final_answer": "",
        "search_messages": [], "planner_messages": [],
    })
    for key in ["tasks", "agent_results", "final_answer"]:
        print(f"{key}:", result[key])
    print("search_messages:", [m.content for m in result["search_messages"]])
    print("planner_messages:", [m.content for m in result["planner_messages"]])


# ---------------------------------------------------------------------------
# 15. REAL SPECIALIST AGENTS CALLED BY A LANGGRAPH WORKER.
# REQUIRES OPENAI_API_KEY. Separate from Example 9's supervisor-as-tool.
# INPUT: Two tasks. OUTPUT: Agent answers merged into shared graph state.
# ---------------------------------------------------------------------------
class example_15_State(TypedDict):
    tasks: list[dict]
    agent_results: Annotated[list[dict], operator.add]
    final_answer: str
        
def example_15_agents_inside_workers():
    from langchain_openai import ChatOpenAI
    from langchain.agents import create_agent
    from langchain_core.tools import tool
    from langgraph.graph import StateGraph, START, END
    from langgraph.types import Send

    @tool
    def find_hotels(city: str) -> str:
        """Return a fictitious hotel suggestion for a city."""
        return f"Fictitious hotel in {city}: Central Hotel"

    @tool
    def get_attractions(city: str) -> str:
        """Return illustrative sightseeing suggestions for a city."""
        return "Eiffel Tower, Louvre, Montmartre" if city.lower() == "paris" else "Sample attractions"

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    search_agent = create_agent(model=llm, tools=[find_hotels],
        system_prompt="Hotel specialist. Use find_hotels. Clearly label fictitious information.")
    planner_agent = create_agent(model=llm, tools=[get_attractions],
        system_prompt="Itinerary specialist. Use get_attractions for travel activities.")



    class WorkerState(TypedDict):
        task: dict

    def orchestrator(state: example_15_State):
        return {"tasks": [
            {"agent": "search", "description": "Find a sample hotel in Paris"},
            {"agent": "planner", "description": "Create a sample 3-day Paris itinerary"},
        ]}

    def dispatch(state: example_15_State):
        return [Send("worker", {"task": task}) for task in state["tasks"]]

    def worker(state: WorkerState):
        task = state["task"]
        agent = search_agent if task["agent"] == "search" else planner_agent
        agent_result = agent.invoke({"messages": [{"role": "user", "content": task["description"]}]})
        answer = str(agent_result["messages"][-1].content)
        updates = {"agent_results": [{"agent": task["agent"], "answer": answer}]}
        return updates  # LangGraph merges these into shared State.

    def synthesizer(state: example_15_State):
        by_agent = {r["agent"]: r["answer"] for r in state["agent_results"]}
        return {"final_answer": f"HOTELS:\n{by_agent.get('search')}\n\nITINERARY:\n{by_agent.get('planner')}"}

    b = StateGraph(example_15_State)
    for name, fn in [("orchestrator", orchestrator), ("worker", worker), ("synthesizer", synthesizer)]:
        b.add_node(name, fn)
    b.add_edge(START, "orchestrator")
    b.add_conditional_edges("orchestrator", dispatch, ["worker"])
    b.add_edge("worker", "synthesizer")
    b.add_edge("synthesizer", END)
    print(b.compile().invoke({"tasks": [], "agent_results": [], "final_answer": ""})["final_answer"])


# ---------------------------------------------------------------------------
# 16. TRACE THE ENTIRE GRAPH via stream_mode="updates".
# INPUT: Graph input. OUTPUT: Per-node updates in execution order/steps.
# ---------------------------------------------------------------------------
class example_16_State(TypedDict):
    tasks: list[str]
    agent_results: Annotated[list[str], operator.add]
    final_answer: str

def example_16_graph_tracing():
    from langgraph.graph import StateGraph, START, END
    from langgraph.types import Send



    class WorkerState(TypedDict):
        task: str

    def orchestrator(state: example_16_State):
        return {"tasks": ["hotels", "itinerary"]}

    def dispatch(state: example_16_State):
        return [Send("worker", {"task": t}) for t in state["tasks"]]

    def worker(state: WorkerState):
        return {"agent_results": [f"Finished {state['task']}"]}

    def synthesizer(state: example_16_State):
        return {"final_answer": "; ".join(sorted(state["agent_results"]))}

    b = StateGraph(example_16_State)
    for name, fn in [("orchestrator", orchestrator), ("worker", worker), ("synthesizer", synthesizer)]:
        b.add_node(name, fn)
    b.add_edge(START, "orchestrator")
    b.add_conditional_edges("orchestrator", dispatch, ["worker"])
    b.add_edge("worker", "synthesizer")
    b.add_edge("synthesizer", END)
    graph = b.compile()
    print("Mermaid graph definition:\n", graph.get_graph().draw_mermaid())
    for update in graph.stream({"tasks": [], "agent_results": [], "final_answer": ""}, stream_mode="updates"):
        print("GRAPH UPDATE:", update)
    print("The Python for-loop reads updates; LangGraph executes the workflow.")


# ---------------------------------------------------------------------------
# 17. MULTI-AGENT CONTENT WRITER: one conditional route to one specialist.
# INPUT: Content request. OUTPUT: SEO / social / general writer response.
# ---------------------------------------------------------------------------
class example_17_State(TypedDict):
    user_query: str
    selected_writer: str
    final_answer: str
def example_17_content_writer():
    from langgraph.graph import StateGraph, START, END



    def router(state: example_17_State):
        query = state["user_query"].lower()
        if "seo" in query:
            return "seo"
        if "twitter" in query or "x post" in query:
            return "social"
        return "general"

    def seo(state: example_17_State):
        return {"selected_writer": "seo", "final_answer": "Sample SEO article"}

    def social(state: example_17_State):
        return {"selected_writer": "social", "final_answer": "Sample X post"}

    def general(state: example_17_State):
        return {"selected_writer": "general", "final_answer": "Sample explanation"}

    b = StateGraph(example_17_State)
    b.add_node("seo_writer", seo)
    b.add_node("social_writer", social)
    b.add_node("general_writer", general)
    b.add_conditional_edges(START, router, {
        "seo": "seo_writer", "social": "social_writer", "general": "general_writer"
    })
    for name in ("seo_writer", "social_writer", "general_writer"):
        b.add_edge(name, END)
    graph = b.compile()
    for query in ("Write SEO content about agents", "Write an X post about LangGraph", "Explain AI agents"):
        print(query, "->", graph.invoke({"user_query": query, "selected_writer": "", "final_answer": ""})["final_answer"])


# ---------------------------------------------------------------------------
# 18. TESTING and DEBUGGING: assert logical results, not worker ordering.
# INPUT: Parallel task execution. OUTPUT: PASS or AssertionError.
# ---------------------------------------------------------------------------
class example_18_State(TypedDict):
    tasks: list[str]
    agent_results: Annotated[list[dict], operator.add]
def example_18_testing():
    from langgraph.graph import StateGraph, START, END
    from langgraph.types import Send



    class WorkerState(TypedDict):
        task: str

    def orchestrator(state: example_18_State):
        return {"tasks": ["search", "planner"]}

    def route(state: example_18_State):
        return [Send("worker", {"task": t}) for t in state["tasks"]]

    def worker(state: WorkerState):
        return {"agent_results": [{"agent": state["task"], "result": "done"}]}

    b = StateGraph(example_18_State)
    b.add_node("orchestrator", orchestrator)
    b.add_node("worker", worker)
    b.add_edge(START, "orchestrator")
    b.add_conditional_edges("orchestrator", route, ["worker"])
    b.add_edge("worker", END)
    result = b.compile().invoke({"tasks": [], "agent_results": []})
    agents = {item["agent"] for item in result["agent_results"]}
    assert agents == {"search", "planner"}, f"Unexpected agent set: {agents}"
    assert len(result["agent_results"]) == 2
    print("PASS: Both workers contributed, independent of result ordering.")


# ---------------------------------------------------------------------------
# SELECT ONE EXAMPLE TO RUN AT A TIME.
# ---------------------------------------------------------------------------
EXAMPLE_TO_RUN = 18

EXAMPLES = {
    1: example_01_multi_agent_basics,
    2: example_02_when_multi_agent,
    3: example_03_workflow_patterns,
    4: example_04_frameworks,
    5: example_05_stategraph_nodes_edges,
    6: example_06_typeddict_annotated_reducer,
    7: example_07_conditional_edges,
    8: example_08_checkpointer,
    9: example_09_supervisor_and_subagent,
    10: example_10_agent_tool_loop_streaming,
    11: example_11_orchestrator_worker_sequential,
    12: example_12_parallel_send,
    13: example_13_synthesizing_results,
    14: example_14_complete_travelplanner_state,
    15: example_15_agents_inside_workers,
    16: example_16_graph_tracing,
    17: example_17_content_writer,
    18: example_18_testing,
}

if __name__ == "__main__":

    if EXAMPLE_TO_RUN == "ALL":

        for example_number, example_function in sorted(EXAMPLES.items()):

            print(f"\nCLASS 04 — EXAMPLE {example_number}")
            print("=" * 50)

            example_function()

    else:

        if EXAMPLE_TO_RUN not in EXAMPLES:
            raise ValueError(
                "Choose EXAMPLE_TO_RUN from 1 to 18 or 'ALL'"
            )

        print(f"\nCLASS 04 — EXAMPLE {EXAMPLE_TO_RUN}")
        print("=" * 50)

        EXAMPLES[EXAMPLE_TO_RUN]()

