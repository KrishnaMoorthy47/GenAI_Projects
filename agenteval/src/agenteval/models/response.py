"""Response models."""

from typing import Optional
from pydantic import BaseModel, Field


class AgentResponse(BaseModel):
    question: str
    answer: Optional[str]
    trajectory: list[dict]
    sql_result: Optional[str]
    iterations: int


class ScoreCard(BaseModel):
    task_success: float = Field(..., ge=0.0, le=1.0)
    tool_accuracy: float = Field(..., ge=0.0, le=1.0)
    trajectory_efficiency: float = Field(..., ge=0.0, le=1.0)
    hallucination: float = Field(..., ge=0.0, le=1.0)
    overall: float = Field(..., ge=0.0, le=1.0)


class CaseResult(BaseModel):
    id: str
    question: str
    expected_answer: str
    actual_answer: Optional[str]
    scores: ScoreCard
    difficulty: str
    tags: list[str]
    error: Optional[str] = None


class EvalReport(BaseModel):
    run_id: str
    status: str                  # pending | running | completed | failed
    total_cases: int = 0
    completed_cases: int = 0
    aggregate_scores: Optional[ScoreCard] = None
    case_results: list[CaseResult] = []
    langsmith_url: Optional[str] = None
    error: Optional[str] = None
