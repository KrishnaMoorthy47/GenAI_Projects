from __future__ import annotations

import asyncio
import logging
import secrets
import uuid

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from langgraph.types import Command
from sse_starlette.sse import EventSourceResponse

from finagent.agents.graph import build_graph
from finagent.config import get_settings
from finagent.models.request import ApprovalRequest, ResearchRequest
from finagent.models.response import InvestmentReport, ResearchResponse, StatusResponse
from finagent.security.audit_log import log_research_request
from finagent.security.prompt_guard import scan_for_injection
from finagent.services.checkpointer import get_checkpointer
from finagent.services.streaming import (
    create_stream_queue,
    get_status,
    run_graph_with_streaming,
    set_status,
    sse_generator,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/research", tags=["research"])

# Same-origin, unauthenticated mirror of every route below — lets the
# bundled browser UI drive real research runs without an API key ever
# reaching the client. Not a bypass of the abuse protection: the shared
# RateLimitMiddleware (see main.py) still caps every client by IP
# regardless of which of these two routers they hit. Mounted under /demo
# in main.py, hidden from the OpenAPI schema there.
demo_router = APIRouter(prefix="/research", tags=["research"])


def verify_api_key(x_api_key: str = Header(...)):
    settings = get_settings()
    try:
        # secrets.compare_digest raises ValueError/TypeError on non-ASCII str
        # inputs — treat that as an invalid key rather than a 500.
        valid = secrets.compare_digest(x_api_key, settings.api_key)
    except (TypeError, ValueError):
        valid = False
    if not valid:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return x_api_key


async def _start_research(body: ResearchRequest) -> ResearchResponse:
    """Start an autonomous research session for a stock ticker.

    ``body.query`` is free text that flows unsanitized into the LLM prompt in
    ``web_research_node``/``sentiment_node``, so it's scanned for prompt-injection
    attempts before any graph work starts. Every request is audit-logged
    regardless of outcome; flagged requests are rejected outright rather than
    silently proceeding.
    """
    thread_id = str(uuid.uuid4())

    guard_result = scan_for_injection(body.query)
    log_research_request(
        thread_id=thread_id,
        ticker=body.ticker,
        query_length=len(body.query),
        guard_flagged=guard_result.flagged,
        guard_reason=guard_result.reason,
    )
    if guard_result.flagged:
        logger.warning(
            "Blocked research request: thread_id=%s ticker=%s reason=%s",
            thread_id,
            body.ticker,
            guard_result.reason,
        )
        raise HTTPException(
            status_code=400,
            detail="Query rejected: contains content resembling a prompt-injection attempt.",
        )

    checkpointer = get_checkpointer()
    graph = build_graph(checkpointer)

    initial_state = {
        "ticker": body.ticker,
        "query": body.query,
        "messages": [],
        "web_research": None,
        "financial_data": None,
        "sentiment": None,
        "report": None,
        "next_agent": "supervisor",
        "human_approved": False,
        "thread_id": thread_id,
    }

    config = {"configurable": {"thread_id": thread_id}}

    create_stream_queue(thread_id)

    # Use asyncio.create_task so the task lives on the event loop independently
    # (BackgroundTasks cancel when the request closes)
    asyncio.create_task(
        run_graph_with_streaming(graph, initial_state, config, thread_id),
        name=f"finagent-{thread_id}",
    )

    logger.info("Research started: thread_id=%s ticker=%s", thread_id, body.ticker)

    return ResearchResponse(
        thread_id=thread_id,
        status="started",
        ticker=body.ticker,
        message=f"Research started for {body.ticker}. Stream events at GET /research/{thread_id}/stream",
    )


def _stream_research(thread_id: str):
    """Stream research progress as Server-Sent Events."""
    return EventSourceResponse(sse_generator(thread_id))


async def _get_research_status(thread_id: str) -> StatusResponse:
    """Get the current status of a research session."""
    status = get_status(thread_id)
    if status is None:
        raise HTTPException(status_code=404, detail="Thread not found")

    # Retrieve state from checkpointer to get ticker info
    checkpointer = get_checkpointer()
    graph = build_graph(checkpointer)
    config = {"configurable": {"thread_id": thread_id}}

    try:
        snapshot = await graph.aget_state(config)
        state_values = snapshot.values if snapshot else {}
        ticker = state_values.get("ticker")
        current_step = state_values.get("next_agent")
    except Exception:
        ticker = None
        current_step = None

    return StatusResponse(
        thread_id=thread_id,
        status=status,
        ticker=ticker,
        current_step=current_step,
    )


async def _approve_research(thread_id: str, body: ApprovalRequest) -> ResearchResponse:
    """Resume a paused research session after human review."""
    status = get_status(thread_id)
    if status is None:
        raise HTTPException(status_code=404, detail="Thread not found")
    if status != "awaiting_approval":
        raise HTTPException(
            status_code=400,
            detail=f"Thread is not awaiting approval (current status: {status})",
        )

    checkpointer = get_checkpointer()
    graph = build_graph(checkpointer)
    config = {"configurable": {"thread_id": thread_id}}

    # Resume the interrupted graph with the approval value
    # The command kwarg is separate from config — this is critical
    try:
        result = await graph.ainvoke(
            None,
            config=config,
            command=Command(resume=body.approved),
        )
        set_status(thread_id, "completed" if body.approved else "rejected")
        logger.info(
            "Research %s: thread_id=%s approved=%s",
            "approved" if body.approved else "rejected",
            thread_id,
            body.approved,
        )
    except Exception as exc:
        logger.error("Error resuming graph for thread_id=%s: %s", thread_id, exc)
        raise HTTPException(status_code=500, detail=f"Failed to resume research: {exc}")

    return ResearchResponse(
        thread_id=thread_id,
        status="completed" if body.approved else "rejected",
        message="Research approved and finalized." if body.approved else "Research rejected.",
    )


async def _get_report(thread_id: str) -> InvestmentReport:
    """Retrieve the final investment report for a completed research session."""
    status = get_status(thread_id)
    if status is None:
        raise HTTPException(status_code=404, detail="Thread not found")
    if status not in ("completed", "awaiting_approval"):
        raise HTTPException(
            status_code=400,
            detail=f"Report not ready (status: {status}). Wait for completion or approval.",
        )

    checkpointer = get_checkpointer()
    graph = build_graph(checkpointer)
    config = {"configurable": {"thread_id": thread_id}}

    try:
        snapshot = await graph.aget_state(config)
        state_values = snapshot.values if snapshot else {}
        report_data = state_values.get("report")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve report: {exc}")

    if not report_data:
        raise HTTPException(status_code=404, detail="Report not yet generated")

    return InvestmentReport(**report_data)


# ── Authenticated routes (real API surface) ─────────────────────────────


@router.post("", response_model=ResearchResponse)
async def start_research(body: ResearchRequest, _: str = Depends(verify_api_key)):
    return await _start_research(body)


@router.get("/{thread_id}/stream")
async def stream_research(thread_id: str, _: str = Depends(verify_api_key)):
    return _stream_research(thread_id)


@router.get("/{thread_id}/status", response_model=StatusResponse)
async def get_research_status(thread_id: str, _: str = Depends(verify_api_key)):
    return await _get_research_status(thread_id)


@router.post("/{thread_id}/approve", response_model=ResearchResponse)
async def approve_research(thread_id: str, body: ApprovalRequest, _: str = Depends(verify_api_key)):
    return await _approve_research(thread_id, body)


@router.get("/{thread_id}/report", response_model=InvestmentReport)
async def get_report(thread_id: str, _: str = Depends(verify_api_key)):
    return await _get_report(thread_id)


# ── Demo routes (no auth — see demo_router docstring above) ─────────────


@demo_router.post("", response_model=ResearchResponse)
async def demo_start_research(body: ResearchRequest):
    return await _start_research(body)


@demo_router.get("/{thread_id}/stream")
async def demo_stream_research(thread_id: str):
    return _stream_research(thread_id)


@demo_router.get("/{thread_id}/status", response_model=StatusResponse)
async def demo_get_research_status(thread_id: str):
    return await _get_research_status(thread_id)


@demo_router.post("/{thread_id}/approve", response_model=ResearchResponse)
async def demo_approve_research(thread_id: str, body: ApprovalRequest):
    return await _approve_research(thread_id, body)


@demo_router.get("/{thread_id}/report", response_model=InvestmentReport)
async def demo_get_report(thread_id: str):
    return await _get_report(thread_id)
