"""LangGraph state definition for the SQL agent."""

from typing import Annotated, Optional

from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages
from typing_extensions import TypedDict


class TrajectoryStep(TypedDict):
    type: str        # "tool_call" | "tool_result"
    name: str
    input: str
    output: str


class SQLAgentState(TypedDict):
    question: str
    messages: Annotated[list[BaseMessage], add_messages]
    trajectory: list[TrajectoryStep]
    sql_result: Optional[str]   # raw SQL output for hallucination scoring
    answer: Optional[str]       # final natural language answer
    iterations: int
