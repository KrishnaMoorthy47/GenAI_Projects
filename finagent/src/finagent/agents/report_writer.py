from __future__ import annotations

import json
import logging

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.types import Command

from finagent.agents.state import FinAgentState
from finagent.config import get_llm
from finagent.models.response import InvestmentReport

logger = logging.getLogger(__name__)

REPORT_WRITER_SYSTEM = """You are a senior financial analyst writing a structured investment brief.
Synthesize all research into a comprehensive investment report.
Be specific, data-driven, and balanced. Include actual numbers from the research.
Your recommendation must be one of: BUY | HOLD | SELL | SPECULATIVE_BUY"""


def report_writer_node(state: FinAgentState) -> Command:
    """Synthesize all research into a structured InvestmentReport."""
    logger.info("report_writer_node: generating report for %s", state["ticker"])

    llm = get_llm()
    structured_llm = llm.with_structured_output(InvestmentReport)

    context_parts = [f"Ticker: {state['ticker']}"]
    if state.get("web_research"):
        context_parts.append(f"## Web Research\n{state['web_research'].get('summary', '')}")
    if state.get("financial_data"):
        context_parts.append(f"## Financial Data\n{state['financial_data'].get('summary', '')}")
    if state.get("sentiment"):
        context_parts.append(f"## Sentiment Analysis\n{state['sentiment'].get('summary', '')}")

    context = "\n\n".join(context_parts)

    try:
        report: InvestmentReport = structured_llm.invoke([
            SystemMessage(content=REPORT_WRITER_SYSTEM),
            HumanMessage(
                content=(
                    f"Generate a comprehensive investment brief for {state['ticker']} "
                    f"based on this research:\n\n{context}"
                )
            ),
        ])
        report_dict = report.model_dump()
    except Exception as exc:
        logger.error("report_writer_node: structured output failed: %s", exc)
        report_dict = {
            "ticker": state["ticker"],
            "company_name": state["ticker"],
            "recommendation": "HOLD",
            "confidence_score": 50,
            "executive_summary": f"Analysis for {state['ticker']}. Report generation encountered an error: {exc}",
            "investment_thesis": "Unable to generate structured report.",
            "financial_highlights": state.get("financial_data", {}).get("summary", "N/A"),
            "risks": "See web research for risk factors.",
            "sentiment_summary": state.get("sentiment", {}).get("summary", "N/A"),
            "data_sources": ["yfinance", "SEC EDGAR", "Tavily web search"],
        }

    return Command(
        update={
            "report": report_dict,
            "messages": [
                AIMessage(
                    content=(
                        f"Investment report generated for {state['ticker']}. "
                        f"Recommendation: {report_dict.get('recommendation')}. "
                        f"Awaiting human review."
                    )
                )
            ],
        },
        goto="human_review",
    )
