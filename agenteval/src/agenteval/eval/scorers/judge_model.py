"""Shared Pydantic model for LLM-as-judge scorer responses."""

from pydantic import BaseModel


class JudgeResult(BaseModel):
    score: float
    reason: str
