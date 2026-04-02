from __future__ import annotations

import operator
from typing import Annotated, Optional

from typing_extensions import TypedDict


class ReviewState(TypedDict):
    repo: str
    pr_number: int
    diff: str
    changed_files: list[str]
    # Annotated with operator.add so both parallel agents can append without overwriting
    security_findings: Annotated[list[dict], operator.add]
    quality_findings: Annotated[list[dict], operator.add]
    final_review: Optional[str]
    pr_url: str
    mode: str  # "webhook" | "api"
    github_token: str
