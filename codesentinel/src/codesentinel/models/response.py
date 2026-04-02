from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class Finding(BaseModel):
    id: str
    category: str
    severity: str  # "critical" | "high" | "medium" | "low" | "info"
    description: str
    file: Optional[str] = None
    line: Optional[int] = None
    snippet: Optional[str] = None
    remediation: Optional[str] = None


class ReviewResult(BaseModel):
    review_id: str
    repo: str
    pr_number: int
    status: str  # "completed" | "failed" | "pending"
    security_findings: list[Finding] = Field(default_factory=list)
    quality_findings: list[Finding] = Field(default_factory=list)
    final_review: str = ""
    files_reviewed: list[str] = Field(default_factory=list)
    total_findings: int = 0
    critical_count: int = 0
    high_count: int = 0
    pr_comment_posted: bool = False
    created_at: datetime = Field(default_factory=datetime.utcnow)
    error: Optional[str] = None
