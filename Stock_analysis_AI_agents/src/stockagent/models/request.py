from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class AnalyzeRequest(BaseModel):
    symbol: str = Field(description="Stock ticker symbol (e.g. AAPL, TSLA, TCS.NS)")
    engine: Literal["langgraph", "crewai"] = Field(
        default="crewai",
        description="Analysis engine: 'crewai' for automated, 'langgraph' for human-in-the-loop",
    )

    @field_validator("symbol")
    @classmethod
    def normalise_symbol(cls, v: str) -> str:
        v = v.strip().upper()
        if not v:
            raise ValueError("symbol cannot be empty")
        return v


class ApproveRequest(BaseModel):
    approved: bool = Field(default=True, description="True to approve and finalise, False to reject")
