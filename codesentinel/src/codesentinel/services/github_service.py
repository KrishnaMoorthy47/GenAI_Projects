from __future__ import annotations

import hashlib
import hmac
import logging

import httpx
from fastapi import HTTPException
from github import Github, GithubException
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from codesentinel.config import get_settings

logger = logging.getLogger(__name__)


def verify_webhook_signature(payload_bytes: bytes, signature_header: str) -> bool:
    """Verify GitHub webhook HMAC-SHA256 signature."""
    settings = get_settings()
    if not settings.github_webhook_secret:
        raise HTTPException(status_code=403, detail="Webhook secret not configured")

    if not signature_header or not signature_header.startswith("sha256="):
        return False

    expected = hmac.new(
        settings.github_webhook_secret.encode("utf-8"),
        payload_bytes,
        hashlib.sha256,
    ).hexdigest()

    received = signature_header.removeprefix("sha256=")
    return hmac.compare_digest(expected, received)


def get_github_client(token: str | None = None) -> Github:
    """Get an authenticated GitHub client."""
    settings = get_settings()
    effective_token = token or settings.github_token
    if not effective_token:
        raise ValueError("No GitHub token available. Set GITHUB_TOKEN env var or pass token in request.")
    return Github(effective_token)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    retry=retry_if_exception_type(httpx.HTTPStatusError),
    reraise=True,
)
async def get_pr_diff(repo: str, pr_number: int, token: str | None = None) -> str:
    """Fetch the raw diff for a pull request using the GitHub API."""
    settings = get_settings()
    effective_token = token or settings.github_token

    headers = {
        "Accept": "application/vnd.github.v3.diff",
        "Authorization": f"Bearer {effective_token}",
        "User-Agent": "CodeSentinel/1.0",
    }

    url = f"https://api.github.com/repos/{repo}/pulls/{pr_number}"
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(url, headers=headers)
        resp.raise_for_status()
        return resp.text


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    retry=retry_if_exception_type(GithubException),
    reraise=False,
)
async def post_pr_review(
    repo: str,
    pr_number: int,
    body: str,
    token: str | None = None,
) -> bool:
    """Post a formal review comment on a GitHub PR."""
    settings = get_settings()
    effective_token = token or settings.github_token

    try:
        gh = Github(effective_token)
        gh_repo = gh.get_repo(repo)
        pr = gh_repo.get_pull(pr_number)
        # create_review posts a formal Review (cleaner than a plain issue comment)
        pr.create_review(body=body, event="COMMENT")
        logger.info("Posted review to %s PR #%d", repo, pr_number)
        return True
    except GithubException as exc:
        logger.error("Failed to post review to %s PR #%d: %s", repo, pr_number, exc)
        return False


def get_pr_url(repo: str, pr_number: int) -> str:
    return f"https://github.com/{repo}/pull/{pr_number}"
