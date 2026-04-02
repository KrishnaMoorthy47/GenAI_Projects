from __future__ import annotations

from pydantic import BaseModel, Field


class ResearchRequest(BaseModel):
    ticker: str = Field(
        description="Stock ticker symbol to research (e.g. AAPL, MSFT, GOOGL)",
        min_length=1,
        max_length=10,
    )
    query: str = Field(
        default="",
        description="Optional custom research focus (defaults to general investment analysis)",
    )

    def model_post_init(self, __context) -> None:
        self.ticker = self.ticker.upper().strip()
        if not self.query:
            self.query = (
                f"Provide a comprehensive investment analysis of {self.ticker} "
                "covering growth prospects, financial health, risks, and recommendation."
            )


class ApprovalRequest(BaseModel):
    approved: bool = Field(description="True to approve the report, False to reject")
    notes: str = Field(default="", description="Optional reviewer notes")
