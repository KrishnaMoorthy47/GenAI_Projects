from __future__ import annotations

import asyncio
import logging
import uuid

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from langgraph.types import Command
from sse_starlette.sse import EventSourceResponse

from finagent.agents.graph import build_graph
from finagent.config import get_settings
from finagent.models.request import ApprovalRequest, ResearchRequest
from finagent.models.response import InvestmentReport, ResearchResponse, StatusResponse
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


def verify_api_key(x_api_key: str = Header(...)):
    settings = get_settings()
    if x_api_key != settings.api_key:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return x_api_key


@router.post("", response_model=ResearchResponse)
async def start_research(
    body: ResearchRequest,
    _: str = Depends(verify_api_key),
):
    """Start an autonomous research session for a stock ticker."""
    thread_id = str(uuid.uuid4())
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


@router.get("/{thread_id}/stream")
async def stream_research(
    thread_id: str,
    _: str = Depends(verify_api_key),
):
    """Stream research progress as Server-Sent Events."""
    return EventSourceResponse(sse_generator(thread_id))


@router.get("/{thread_id}/status", response_model=StatusResponse)
async def get_research_status(
    thread_id: str,
    _: str = Depends(verify_api_key),
):
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


@router.post("/{thread_id}/approve", response_model=ResearchResponse)
async def approve_research(
    thread_id: str,
    body: ApprovalRequest,
    _: str = Depends(verify_api_key),
):
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


@router.get("/{thread_id}/report", response_model=InvestmentReport)
async def get_report(
    thread_id: str,
    _: str = Depends(verify_api_key),
):
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
