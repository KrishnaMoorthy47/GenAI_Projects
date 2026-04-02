from __future__ import annotations

import asyncio
import json
import logging

from fastapi import APIRouter, Header, HTTPException, Request

from codesentinel.api.review import _run_review
from codesentinel.services import github_service, review_store

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/webhook", tags=["webhook"])

# PR actions that trigger a review
REVIEW_ACTIONS = {"opened", "synchronize", "reopened"}


@router.post("/github")
async def github_webhook(
    request: Request,
    x_hub_signature_256: str = Header(default=""),
    x_github_event: str = Header(default=""),
):
    """Receive and process GitHub webhook events."""
    payload_bytes = await request.body()

    # Verify HMAC signature
    if not github_service.verify_webhook_signature(payload_bytes, x_hub_signature_256):
        raise HTTPException(status_code=403, detail="Invalid webhook signature")

    # Only handle pull_request events
    if x_github_event != "pull_request":
        return {"status": "ignored", "event": x_github_event}

    try:
        payload = json.loads(payload_bytes)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON payload")

    action = payload.get("action", "")
    if action not in REVIEW_ACTIONS:
        return {"status": "ignored", "action": action}

    pr_data = payload.get("pull_request", {})
    repo_data = payload.get("repository", {})

    pr_number = pr_data.get("number")
    repo = repo_data.get("full_name")

    if not pr_number or not repo:
        raise HTTPException(status_code=400, detail="Missing pr_number or repo in payload")

    # Create review record and kick off async task
    review_id = await review_store.create_review(repo, pr_number)

    asyncio.create_task(
        _run_review(review_id, repo, pr_number, token=None),
        name=f"codesentinel-webhook-{review_id}",
    )

    logger.info(
        "Webhook triggered review %s for %s PR #%d (action: %s)",
        review_id, repo, pr_number, action,
    )

    return {
        "status": "accepted",
        "review_id": review_id,
        "repo": repo,
        "pr_number": pr_number,
    }
