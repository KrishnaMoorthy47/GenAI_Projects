"""Request models."""

from typing import Optional
from pydantic import BaseModel, Field


class AgentQueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Natural language question about the database")


class EvalRunRequest(BaseModel):
    dataset_path: Optional[str] = Field(
        None, description="Path to eval_cases.json (defaults to datasets/eval_cases.json)"
    )
    experiment_prefix: Optional[str] = Field(None, description="LangSmith experiment prefix override")
    push_to_langsmith: bool = Field(True, description="Whether to push dataset and results to LangSmith")
