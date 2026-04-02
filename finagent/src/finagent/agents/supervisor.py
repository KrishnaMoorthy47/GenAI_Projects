from __future__ import annotations

import logging
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.types import Command

from finagent.agents.state import FinAgentState
from finagent.config import get_llm

logger = logging.getLogger(__name__)

VALID_NEXT_NODES = {"web_research", "financial_data", "sentiment", "report_writer"}

SUPERVISOR_SYSTEM = """You are a financial research supervisor orchestrating a team of specialist agents.

Given the research task and the work completed so far, decide which agent to call next.

Available agents:
- web_research: Search recent news, analyst reports, and market sentiment for the ticker
- financial_data: Retrieve current stock price, financials, earnings history, and SEC filings
- sentiment: Analyze earnings call transcripts and qualitative signals
- report_writer: Compile all research into a structured investment brief (call LAST, only when all research is complete)

Rules:
1. Always start with web_research AND financial_data before sentiment.
2. Call sentiment only after web_research and financial_data are done.
3. Call report_writer only when web_research, financial_data, AND sentiment are all complete.
4. Return ONLY the agent name — no explanation, no punctuation.

Respond with exactly one of: web_research | financial_data | sentiment | report_writer"""


def supervisor_node(state: FinAgentState) -> Command[Literal[
    "web_research", "financial_data", "sentiment", "report_writer"
]]:
    """Route to the next agent based on what research has been completed."""
    web_done = state.get("web_research") is not None
    fin_done = state.get("financial_data") is not None
    sent_done = state.get("sentiment") is not None

    # Fast-path routing — avoid LLM call when deterministic
    if not web_done:
        logger.info("Supervisor → web_research")
        return Command(goto="web_research")
    if not fin_done:
        logger.info("Supervisor → financial_data")
        return Command(goto="financial_data")
    if not sent_done:
        logger.info("Supervisor → sentiment")
        return Command(goto="sentiment")
    if web_done and fin_done and sent_done:
        logger.info("Supervisor → report_writer (all research complete)")
        return Command(goto="report_writer")

    # Fallback: ask LLM when state is ambiguous
    llm = get_llm()
    status_summary = (
        f"Ticker: {state['ticker']}\n"
        f"Web research done: {web_done}\n"
        f"Financial data done: {fin_done}\n"
        f"Sentiment done: {sent_done}\n"
    )
    response = llm.invoke([
        SystemMessage(content=SUPERVISOR_SYSTEM),
        HumanMessage(content=status_summary),
    ])

    raw = response.content.strip().lower().replace("-", "_")
    # Validate LLM output — never pass unknown node names into the graph
    if raw not in VALID_NEXT_NODES:
        logger.warning("Supervisor LLM returned unknown node '%s'; falling back to web_research", raw)
        raw = "web_research"

    logger.info("Supervisor (LLM) → %s", raw)
    return Command(goto=raw)
