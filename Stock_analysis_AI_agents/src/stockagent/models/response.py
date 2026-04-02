from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class AnalyzeResponse(BaseModel):
    """Returned when an analysis job is started or polled."""

    job_id: str = Field(description="Unique ID for the analysis job")
    engine: str = Field(description="langgraph | crewai")
    status: str = Field(description="started | pending | running | completed | failed | awaiting_approval | rejected")
    awaiting_approval: bool = Field(default=False, description="True when a LangGraph job needs human approval")
    result: Optional[Any] = Field(default=None, description="Final analysis text (available when status=completed)")


class ApproveResponse(BaseModel):
    """Returned after approving or rejecting a LangGraph job."""

    job_id: str
    engine: str
    status: str = Field(description="completed | rejected")


class StockQuoteResponse(BaseModel):
    """Real-time stock quote data."""

    symbol: str
    name: Optional[str] = None
    price: Optional[float] = None
    change_pct: Optional[float] = Field(default=None, description="Day change in percent")
    market_cap: Optional[int] = None
    pe_ratio: Optional[float] = None
    week_52_high: Optional[float] = Field(default=None, alias="52w_high")
    week_52_low: Optional[float] = Field(default=None, alias="52w_low")
    sector: Optional[str] = None
    currency: Optional[str] = None

    model_config = {"populate_by_name": True}
