from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class JobSearchStartResponse(BaseModel):
    """Returned immediately when a job search is kicked off."""

    search_id: str = Field(description="Unique ID to poll for status and results")
    status: str = Field(description="Initial status: pending")
    companies: list[str] = Field(description="Companies being searched")
    role: str = Field(description="Job role being searched for")
    cv_chars: int = Field(description="Number of characters extracted from the uploaded CV")


class JobSearchStatusResponse(BaseModel):
    """Returned when polling the status of a running search."""

    search_id: str
    status: str = Field(description="pending | running | completed | failed")
    companies: list[str]
    role: str
    progress: dict[str, Any] = Field(default_factory=dict, description="Per-company progress map")
    jobs_found: int = Field(description="Number of job matches found so far")
    error: Optional[str] = None


class JobResult(BaseModel):
    """A single scored job listing."""

    company: str
    title: str
    url: Optional[str] = None
    fit_score: Optional[int] = Field(default=None, ge=0, le=100, description="LLM fit score 0-100")
    summary: Optional[str] = None


class JobSearchResultsResponse(BaseModel):
    """Returned when a search is completed with ranked results."""

    search_id: str
    status: str
    role: str
    jobs: list[dict[str, Any]] = Field(description="Ranked job listings")
    total: int = Field(description="Total number of job matches found")
