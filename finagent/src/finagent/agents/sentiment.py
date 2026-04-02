from __future__ import annotations

import logging

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.types import Command

from finagent.agents.state import FinAgentState
from finagent.config import get_llm
from finagent.tools.web_search import web_search

logger = logging.getLogger(__name__)

SENTIMENT_SYSTEM = """You are a financial sentiment analyst specializing in qualitative signals.
Analyze the following research data and search for additional qualitative information:
- Management tone and forward guidance from earnings calls
- Institutional investor sentiment and insider trading activity
- Social/Reddit/analyst community sentiment
- ESG and governance factors

Use web_search to find recent earnings call highlights or management commentary.
Return a structured sentiment assessment with:
- Overall sentiment: Bullish | Neutral | Bearish (with confidence 0-100)
- Key positive signals
- Key negative signals / risks
- Management credibility assessment
- Recommended investor type (growth/value/dividend/speculative)"""


def sentiment_node(state: FinAgentState) -> Command:
    """Analyze qualitative signals, earnings tone, and market sentiment."""
    logger.info("sentiment_node: analyzing sentiment for %s", state["ticker"])

    llm = get_llm()
    llm_with_tools = llm.bind_tools([web_search])

    context_parts = [f"Ticker: {state['ticker']}"]
    if state.get("web_research"):
        context_parts.append(f"Web Research:\n{state['web_research'].get('summary', '')}")
    if state.get("financial_data"):
        context_parts.append(f"Financial Data:\n{state['financial_data'].get('summary', '')}")

    context = "\n\n".join(context_parts)

    messages = [
        SystemMessage(content=SENTIMENT_SYSTEM),
        HumanMessage(
            content=(
                f"Perform sentiment analysis for {state['ticker']} using this research:\n\n{context}\n\n"
                "Search for any recent earnings call highlights or management commentary."
            )
        ),
    ]

    for _ in range(4):
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
            "sentiment": {"summary": summary, "ticker": state["ticker"]},
            "messages": [AIMessage(content=f"Sentiment analysis complete for {state['ticker']}:\n{summary}")],
        },
        goto="supervisor",
    )
