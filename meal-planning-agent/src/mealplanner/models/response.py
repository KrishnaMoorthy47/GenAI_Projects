from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class PlanStartResponse(BaseModel):
    """Returned immediately when a meal plan generation is kicked off."""

    plan_id: str = Field(description="Unique ID to poll for status and results")
    status: str = Field(description="Initial status: pending")


class PlanStatusResponse(BaseModel):
    """Returned when polling the status of a running meal plan generation."""

    plan_id: str
    status: str = Field(description="pending | running | completed | failed")
    step: Optional[str] = Field(default=None, description="Current pipeline step name")
    error: Optional[str] = None
