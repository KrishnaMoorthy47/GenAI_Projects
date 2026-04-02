from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class ReviewRequest(BaseModel):
    repo: str = Field(
        description="GitHub repository in 'owner/repo' format (e.g. 'octocat/Hello-World')"
    )
    pr_number: int = Field(description="Pull request number", gt=0)
    github_token: Optional[str] = Field(
        default=None,
        description="GitHub personal access token. Falls back to GITHUB_TOKEN env var.",
    )


class WebhookPayload(BaseModel):
    """Minimal GitHub PR webhook payload — only fields we need."""
    action: str
    number: Optional[int] = None
    pull_request: Optional[dict] = None
    repository: Optional[dict] = None
