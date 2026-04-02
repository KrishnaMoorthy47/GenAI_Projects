from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, Depends, Header, HTTPException

from codesentinel.agents.graph import get_review_graph
from codesentinel.config import get_settings
from codesentinel.models.request import ReviewRequest
from codesentinel.models.response import Finding, ReviewResult
from codesentinel.services import github_service, review_store

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/review", tags=["review"])


def verify_api_key(x_api_key: str = Header(...)):
    settings = get_settings()
    if x_api_key != settings.api_key:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return x_api_key


async def _run_review(review_id: str, repo: str, pr_number: int, token: str | None) -> None:
    """Background task: fetch diff → run graph → post comment → update DB."""
    try:
        # Fetch diff
        diff = await github_service.get_pr_diff(repo, pr_number, token)
        pr_url = github_service.get_pr_url(repo, pr_number)

        # Run the review graph
        graph = get_review_graph()
        initial_state = {
            "repo": repo,
            "pr_number": pr_number,
            "diff": diff,
            "changed_files": [],
            "security_findings": [],
            "quality_findings": [],
            "final_review": None,
            "pr_url": pr_url,
            "mode": "api",
            "github_token": token or "",
        }

        result_state = await graph.ainvoke(initial_state)

        security_findings = result_state.get("security_findings", [])
        quality_findings = result_state.get("quality_findings", [])
        final_review = result_state.get("final_review", "")
        changed_files = result_state.get("changed_files", [])

        critical_count = sum(1 for f in security_findings if f.get("severity") == "critical")
        high_count = sum(
            1 for f in security_findings + quality_findings if f.get("severity") == "high"
        )

        # Post review comment to GitHub PR
        comment_posted = False
        if final_review:
            comment_posted = await github_service.post_pr_review(repo, pr_number, final_review, token)

        # Persist to DB
        await review_store.update_review(
            review_id,
            status="completed",
            security_findings=security_findings,
            quality_findings=quality_findings,
            final_review=final_review,
            files_reviewed=changed_files,
            total_findings=len(security_findings) + len(quality_findings),
            critical_count=critical_count,
            high_count=high_count,
            pr_comment_posted=int(comment_posted),
        )

        logger.info(
            "Review %s completed: %d security, %d quality findings",
            review_id,
            len(security_findings),
            len(quality_findings),
        )

    except Exception as exc:
        logger.error("Review %s failed: %s", review_id, exc)
        await review_store.update_review(review_id, status="failed", error=str(exc))


@router.post("", response_model=ReviewResult)
async def trigger_review(
    body: ReviewRequest,
    _: str = Depends(verify_api_key),
):
    """Trigger a PR review. Returns immediately; review runs in the background."""
    review_id = await review_store.create_review(body.repo, body.pr_number)

    asyncio.create_task(
        _run_review(review_id, body.repo, body.pr_number, body.github_token),
        name=f"codesentinel-{review_id}",
    )

    logger.info("Review started: %s for %s PR #%d", review_id, body.repo, body.pr_number)

    return ReviewResult(
        review_id=review_id,
        repo=body.repo,
        pr_number=body.pr_number,
        status="pending",
        final_review="Review in progress...",
    )


@router.get("/{review_id}", response_model=ReviewResult)
async def get_review(
    review_id: str,
    _: str = Depends(verify_api_key),
):
    """Get the status and results of a review."""
    record = await review_store.get_review(review_id)
    if not record:
        raise HTTPException(status_code=404, detail="Review not found")

    return ReviewResult(
        review_id=record["review_id"],
        repo=record["repo"],
        pr_number=record["pr_number"],
        status=record["status"],
        security_findings=[Finding(**f) for f in record["security_findings"] if isinstance(f, dict)],
        quality_findings=[Finding(**f) for f in record["quality_findings"] if isinstance(f, dict)],
        final_review=record["final_review"],
        files_reviewed=record["files_reviewed"],
        total_findings=record["total_findings"],
        critical_count=record["critical_count"],
        high_count=record["high_count"],
        pr_comment_posted=record["pr_comment_posted"],
        created_at=record["created_at"],
        error=record["error"],
    )
