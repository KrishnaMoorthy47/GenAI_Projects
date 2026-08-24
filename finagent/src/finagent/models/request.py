from __future__ import annotations

import re

from pydantic import BaseModel, Field

# Excludes \t/\n/\r (0x09/0x0a/0x0d) — multi-line research queries are legitimate.
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


class ResearchRequest(BaseModel):
    ticker: str = Field(
        description="Stock ticker symbol to research (e.g. AAPL, MSFT, GOOGL)",
        min_length=1,
        max_length=10,
    )
    query: str = Field(
        default="",
        max_length=2000,
        description="Optional custom research focus (defaults to general investment analysis)",
    )

    def model_post_init(self, __context) -> None:
        self.ticker = self.ticker.upper().strip()
        self.query = _CONTROL_CHAR_RE.sub("", self.query)
        if not self.query:
            self.query = (
                f"Provide a comprehensive investment analysis of {self.ticker} "
                "covering growth prospects, financial health, risks, and recommendation."
            )


class ApprovalRequest(BaseModel):
    approved: bool = Field(description="True to approve the report, False to reject")
    notes: str = Field(default="", description="Optional reviewer notes")
