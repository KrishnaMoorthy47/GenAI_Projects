"""Agent nodes: agent_node (ReAct loop) and tool_node."""

import json
import logging
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.prebuilt import ToolNode

from tenacity import retry, stop_after_attempt, wait_exponential

from agenteval.agent.state import SQLAgentState, TrajectoryStep
from agenteval.agent.tools import get_schema, list_tables, run_query
from agenteval.config import get_llm, get_settings

logger = logging.getLogger(__name__)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8), reraise=True)
def _invoke_llm(llm, messages):
    return llm.invoke(messages)

TOOLS = [list_tables, get_schema, run_query]
TOOL_NODE = ToolNode(TOOLS)

SYSTEM_PROMPT = """You are a SQL expert assistant. Answer the user's question by querying a SQLite database.

Always follow this process:
1. Call list_tables to see available tables.
2. Call get_schema for relevant tables to understand their structure.
3. Write and call run_query with a correct SELECT statement.
4. Provide a concise natural language answer based on the query result.

Be precise and only include facts supported by query results."""


def make_agent_node(llm_with_tools):
    """Return an agent_node closure bound to the given LLM."""

    def agent_node(state: SQLAgentState) -> dict[str, Any]:
        settings = get_settings()
        iterations = state.get("iterations", 0)

        if iterations >= settings.max_agent_iterations:
            logger.warning("Max iterations (%d) reached", settings.max_agent_iterations)
            last = state["messages"][-1] if state["messages"] else None
            answer = last.content if last else "Could not determine an answer within iteration limit."
            return {"answer": answer, "iterations": iterations}

        messages = state["messages"]
        response: AIMessage = _invoke_llm(llm_with_tools, messages)

        trajectory: list[TrajectoryStep] = list(state.get("trajectory", []))
        sql_result = state.get("sql_result")

        # Record tool calls in trajectory
        if response.tool_calls:
            for tc in response.tool_calls:
                trajectory.append(
                    TrajectoryStep(
                        type="tool_call",
                        name=tc["name"],
                        input=json.dumps(tc["args"]),
                        output="",
                    )
                )

        updates: dict[str, Any] = {
            "messages": [response],
            "trajectory": trajectory,
            "iterations": iterations + 1,
        }

        # If no tool calls, extract final answer
        if not response.tool_calls:
            updates["answer"] = response.content

        return updates

    return agent_node


def tool_node(state: SQLAgentState) -> dict[str, Any]:
    """Execute tool calls and capture sql_result for hallucination scoring."""
    result = TOOL_NODE.invoke(state)

    # Update trajectory with tool outputs
    trajectory: list[TrajectoryStep] = list(state.get("trajectory", []))
    sql_result = state.get("sql_result")
    new_messages = result.get("messages", [])

    # Match ToolMessages back to trajectory entries
    tool_call_idx = len([s for s in trajectory if s["type"] == "tool_call"]) - len(
        [m for m in new_messages if isinstance(m, ToolMessage)]
    )

    for msg in new_messages:
        if isinstance(msg, ToolMessage):
            if tool_call_idx < len(trajectory):
                trajectory[tool_call_idx] = TrajectoryStep(
                    type=trajectory[tool_call_idx]["type"],
                    name=trajectory[tool_call_idx]["name"],
                    input=trajectory[tool_call_idx]["input"],
                    output=msg.content,
                )
                # Capture run_query output for hallucination scoring
                if trajectory[tool_call_idx]["name"] == "run_query":
                    sql_result = msg.content
            tool_call_idx += 1

    updates = {**result, "trajectory": trajectory}
    if sql_result is not None:
        updates["sql_result"] = sql_result
    return updates
