from __future__ import annotations

import logging

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.types import Command

from finagent.agents.state import FinAgentState
from finagent.config import get_llm
from finagent.tools.calculator import calculator
from finagent.tools.finance import get_earnings_history, get_sec_filings, get_stock_info

logger = logging.getLogger(__name__)

FINANCIAL_DATA_SYSTEM = """You are a financial data analyst.
Use the available tools to retrieve comprehensive financial data for the given stock ticker.
Always call:
1. get_stock_info — for price, valuation, and fundamentals
2. get_earnings_history — for recent earnings trend
3. get_sec_filings — for latest annual (10-K) filing reference

Use the calculator tool to compute derived metrics if needed (e.g., revenue growth rate).
Summarize all retrieved data in a structured financial profile."""


def financial_data_node(state: FinAgentState) -> Command:
    """Retrieve stock fundamentals, earnings history, and SEC filings."""
    logger.info("financial_data_node: fetching data for %s", state["ticker"])

    llm = get_llm()
    tools = [get_stock_info, get_earnings_history, get_sec_filings, calculator]
    llm_with_tools = llm.bind_tools(tools)

    tool_map = {
        "get_stock_info": get_stock_info,
        "get_earnings_history": get_earnings_history,
        "get_sec_filings": get_sec_filings,
        "calculator": calculator,
    }

    messages = [
        SystemMessage(content=FINANCIAL_DATA_SYSTEM),
        HumanMessage(
            content=f"Retrieve comprehensive financial data for: {state['ticker']}"
        ),
    ]

    for _ in range(8):
        response = llm_with_tools.invoke(messages)
        messages.append(response)

        if not response.tool_calls:
            break

        for tool_call in response.tool_calls:
            fn = tool_map.get(tool_call["name"])
            if fn:
                tool_result = fn.invoke(tool_call["args"])
            else:
                tool_result = f"Unknown tool: {tool_call['name']}"

            messages.append(
                ToolMessage(
                    content=str(tool_result),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
            )

    summary = response.content if hasattr(response, "content") else str(response)

    return Command(
        update={
            "financial_data": {"summary": summary, "ticker": state["ticker"]},
            "messages": [AIMessage(content=f"Financial data complete for {state['ticker']}:\n{summary}")],
        },
        goto="supervisor",
    )
