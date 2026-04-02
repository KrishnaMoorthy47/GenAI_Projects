"""Build the SQL agent LangGraph graph."""

import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.memory import MemorySaver

from agenteval.agent.nodes import TOOLS, make_agent_node, tool_node
from agenteval.agent.state import SQLAgentState
from agenteval.config import get_llm

logger = logging.getLogger(__name__)


def _should_continue(state: SQLAgentState) -> str:
    """Route: if last message has tool calls → tool_node, else → END."""
    messages = state.get("messages", [])
    if not messages:
        return END
    last = messages[-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tool_node"
    return END


def build_sql_agent(llm=None) -> Any:
    """Build and compile the SQL agent graph.

    Args:
        llm: Optional LLM instance (defaults to get_llm()). Pass a mock in tests.

    Returns:
        Compiled LangGraph graph.
    """
    if llm is None:
        llm = get_llm()

    llm_with_tools = llm.bind_tools(TOOLS)
    agent_node = make_agent_node(llm_with_tools)

    graph = StateGraph(SQLAgentState)
    graph.add_node("agent_node", agent_node)
    graph.add_node("tool_node", tool_node)

    graph.add_edge(START, "agent_node")
    graph.add_conditional_edges("agent_node", _should_continue, ["tool_node", END])
    graph.add_edge("tool_node", "agent_node")

    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)


async def run_sql_agent(question: str, llm=None) -> SQLAgentState:
    """Run the SQL agent for a single question and return final state."""
    from langchain_core.messages import HumanMessage, SystemMessage
    from agenteval.agent.nodes import SYSTEM_PROMPT
    import uuid

    graph = build_sql_agent(llm=llm)
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    initial_state: SQLAgentState = {
        "question": question,
        "messages": [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=question),
        ],
        "trajectory": [],
        "sql_result": None,
        "answer": None,
        "iterations": 0,
    }

    final_state = await graph.ainvoke(initial_state, config=config)
    return final_state
