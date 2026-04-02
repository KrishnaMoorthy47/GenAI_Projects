from __future__ import annotations

from typing import Annotated, Optional

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class FinAgentState(TypedDict):
    ticker: str
    query: str
    messages: Annotated[list[BaseMessage], add_messages]
    web_research: Optional[dict]
    financial_data: Optional[dict]
    sentiment: Optional[dict]
    report: Optional[dict]
    next_agent: str
    human_approved: bool
    thread_id: str
