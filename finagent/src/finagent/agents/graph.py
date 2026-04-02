from __future__ import annotations

import logging

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import StateGraph
from langgraph.types import interrupt

from finagent.agents.financial_data import financial_data_node
from finagent.agents.report_writer import report_writer_node
from finagent.agents.sentiment import sentiment_node
from finagent.agents.state import FinAgentState
from finagent.agents.supervisor import supervisor_node
from finagent.agents.web_research import web_research_node

logger = logging.getLogger(__name__)


def human_review_node(state: FinAgentState) -> dict:
    """Pause execution and wait for human approval of the investment report."""
    report = state.get("report", {})
    ticker = state.get("ticker", "unknown")

    # interrupt() suspends the graph and returns the value to the caller
    # Graph resumes when POST /research/{id}/approve sends Command(resume=...)
    approval_value = interrupt({
        "message": (
            f"Investment report for {ticker} is ready. "
            f"Recommendation: {report.get('recommendation', 'N/A')}. "
            "Please review and approve or reject."
        ),
        "report_preview": {
            "ticker": ticker,
            "recommendation": report.get("recommendation"),
            "confidence_score": report.get("confidence_score"),
            "executive_summary": report.get("executive_summary"),
        },
    })

    approved = approval_value if isinstance(approval_value, bool) else bool(approval_value)
    logger.info("human_review_node: approved=%s for ticker=%s", approved, ticker)

    return {"human_approved": approved}


def build_graph(checkpointer: BaseCheckpointSaver) -> StateGraph:
    """Build and compile the FinAgent LangGraph workflow."""
    builder = StateGraph(FinAgentState)

    # Register nodes
    builder.add_node("supervisor", supervisor_node)
    builder.add_node("web_research", web_research_node)
    builder.add_node("financial_data", financial_data_node)
    builder.add_node("sentiment", sentiment_node)
    builder.add_node("report_writer", report_writer_node)
    builder.add_node("human_review", human_review_node)

    # Entry point
    builder.set_entry_point("supervisor")

    # human_review → END (graph finishes after human decision)
    builder.set_finish_point("human_review")

    # All routing is done via Command(goto=...) inside the nodes themselves
    # (supervisor routes to workers; workers route back to supervisor;
    # report_writer routes to human_review)
    # No add_conditional_edges needed.

    graph = builder.compile(checkpointer=checkpointer, interrupt_before=[])
    logger.info("FinAgent graph compiled successfully")
    return graph
