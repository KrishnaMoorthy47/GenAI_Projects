from __future__ import annotations

import logging

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.types import Command

from finagent.agents.state import FinAgentState
from finagent.config import get_llm
from finagent.tools.web_search import web_search

logger = logging.getLogger(__name__)

WEB_RESEARCH_SYSTEM = """You are a financial web research specialist.
Search for recent news, analyst opinions, market trends, and competitive landscape for the given stock ticker.
Use the web_search tool to find relevant information. Make 2-3 targeted searches.
Summarize your findings in a structured format covering:
- Recent news and events
- Analyst sentiment and price targets
- Competitive position
- Key risks and opportunities"""


def web_research_node(state: FinAgentState) -> Command:
    """Search web for company news, analyst reports, and market sentiment."""
    logger.info("web_research_node: researching %s", state["ticker"])

    llm = get_llm()
    llm_with_tools = llm.bind_tools([web_search])

    messages = [
        SystemMessage(content=WEB_RESEARCH_SYSTEM),
        HumanMessage(content=f"Research the stock: {state['ticker']}\nQuery: {state['query']}"),
    ]

    # Agentic tool-call loop (max 5 iterations to prevent runaway)
    for _ in range(5):
        response = llm_with_tools.invoke(messages)
        messages.append(response)

        if not response.tool_calls:
            break

        for tool_call in response.tool_calls:
            tool_result = web_search.invoke(tool_call["args"])
            messages.append(
                ToolMessage(
                    content=str(tool_result),
                    name="web_search",
                    tool_call_id=tool_call["id"],
                )
            )

    summary = response.content if hasattr(response, "content") else str(response)

    return Command(
        update={
            "web_research": {"summary": summary, "sources": "web"},
            "messages": [AIMessage(content=f"Web research complete for {state['ticker']}:\n{summary}")],
        },
        goto="supervisor",
    )
