from __future__ import annotations

import asyncio
import logging
import uuid

import yfinance as yf
from fastapi import APIRouter, Depends, HTTPException, Security
from fastapi.security.api_key import APIKeyHeader
from langgraph.types import Command

from app.agents.graph import build_graph
from app.services.checkpointer import get_checkpointer
from app.services.streaming import (
    create_stream_queue,
    get_status,
    run_graph_with_streaming,
    set_status,
)
from stockagent.config import get_settings
from stockagent.models.request import AnalyzeRequest, ApproveRequest
from stockagent.models.response import AnalyzeResponse, ApproveResponse, StockQuoteResponse
from stockagent.services.crew import create_job, get_job_store, run_analysis

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1")

_api_key_header = APIKeyHeader(name="x-api-key", auto_error=True)

# Tracks which engine is behind each job_id
_job_meta: dict[str, str] = {}  # job_id → "langgraph" | "crewai"

POPULAR_STOCKS = [
    {"symbol": "AAPL",        "name": "Apple Inc.",                "exchange": "NASDAQ"},
    {"symbol": "MSFT",        "name": "Microsoft Corp.",           "exchange": "NASDAQ"},
    {"symbol": "GOOGL",       "name": "Alphabet Inc.",             "exchange": "NASDAQ"},
    {"symbol": "AMZN",        "name": "Amazon.com Inc.",           "exchange": "NASDAQ"},
    {"symbol": "NVDA",        "name": "NVIDIA Corp.",              "exchange": "NASDAQ"},
    {"symbol": "META",        "name": "Meta Platforms Inc.",       "exchange": "NASDAQ"},
    {"symbol": "TSLA",        "name": "Tesla Inc.",                "exchange": "NASDAQ"},
    {"symbol": "TCS.NS",      "name": "TCS (NSE)",                 "exchange": "NSE"},
    {"symbol": "INFY.NS",     "name": "Infosys (NSE)",             "exchange": "NSE"},
    {"symbol": "RELIANCE.NS", "name": "Reliance Industries (NSE)", "exchange": "NSE"},
]


def verify_key(key: str = Security(_api_key_header)) -> str:
    if key != get_settings().api_key:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return key


@router.post("/analyze", dependencies=[Depends(verify_key)], response_model=AnalyzeResponse)
async def analyze_stock(body: AnalyzeRequest) -> AnalyzeResponse:
    """
    Kick off a stock analysis.

    - engine="crewai"    → fully automated 4-agent CrewAI crew, poll /analyze/{job_id}
    - engine="langgraph" → LangGraph graph with human-in-the-loop; poll /analyze/{job_id},
                           then POST /analyze/{job_id}/approve when status=awaiting_approval
    """
    symbol = body.symbol.strip().upper()
    if not symbol:
        raise HTTPException(status_code=400, detail="symbol cannot be empty")

    if body.engine == "langgraph":
        thread_id = str(uuid.uuid4())
        checkpointer = get_checkpointer()
        graph = build_graph(checkpointer)
        initial_state = {
            "ticker": symbol,
            "query": f"Provide a full investment analysis for {symbol}",
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
        asyncio.create_task(
            run_graph_with_streaming(graph, initial_state, config, thread_id),
            name=f"stockintel-lg-{thread_id}",
        )
        _job_meta[thread_id] = "langgraph"
        return AnalyzeResponse(job_id=thread_id, engine="langgraph", status="started")

    else:
        job_id = create_job(symbol)
        asyncio.create_task(
            run_analysis(job_id, f"Provide a full investment analysis for {symbol}"),
            name=f"stockintel-crew-{job_id}",
        )
        _job_meta[job_id] = "crewai"
        return AnalyzeResponse(job_id=job_id, engine="crewai", status="pending")


@router.get("/analyze/{job_id}", dependencies=[Depends(verify_key)], response_model=AnalyzeResponse)
async def get_analysis(job_id: str) -> AnalyzeResponse:
    """Poll status and result of an analysis job (both engines)."""
    engine = _job_meta.get(job_id)
    if not engine:
        raise HTTPException(status_code=404, detail="Job not found")

    if engine == "langgraph":
        status = get_status(job_id)
        if status is None:
            raise HTTPException(status_code=404, detail="Job not found")

        result = None
        if status in ("completed", "awaiting_approval"):
            try:
                graph = build_graph(get_checkpointer())
                config = {"configurable": {"thread_id": job_id}}
                snapshot = await graph.aget_state(config)
                report_data = snapshot.values.get("report") if snapshot else None
                if report_data:
                    result = report_data.get("content") or str(report_data)
            except Exception as exc:
                logger.warning("Could not fetch report from state for %s: %s", job_id, exc)

        return AnalyzeResponse(
            job_id=job_id,
            engine="langgraph",
            status=status,
            awaiting_approval=status == "awaiting_approval",
            result=result,
        )

    else:  # crewai
        store = get_job_store()
        entry = store.get(job_id)
        if not entry:
            raise HTTPException(status_code=404, detail="Job not found")
        return AnalyzeResponse(
            job_id=job_id,
            engine="crewai",
            status=entry["status"],
            awaiting_approval=False,
            result=entry.get("result"),
        )


@router.post("/analyze/{job_id}/approve", dependencies=[Depends(verify_key)], response_model=ApproveResponse)
async def approve_analysis(job_id: str, body: ApproveRequest) -> ApproveResponse:
    """Resume a LangGraph job that is awaiting human approval."""
    engine = _job_meta.get(job_id)
    if not engine:
        raise HTTPException(status_code=404, detail="Job not found")
    if engine != "langgraph":
        raise HTTPException(status_code=400, detail="Approve only applies to LangGraph jobs")

    status = get_status(job_id)
    if status != "awaiting_approval":
        raise HTTPException(
            status_code=400,
            detail=f"Job is not awaiting approval (current status: {status})",
        )

    graph = build_graph(get_checkpointer())
    config = {"configurable": {"thread_id": job_id}}

    try:
        await graph.ainvoke(None, config=config, command=Command(resume=body.approved))
        new_status = "completed" if body.approved else "rejected"
        set_status(job_id, new_status)
        logger.info("LangGraph job %s → %s", job_id, new_status)
    except Exception as exc:
        logger.error("Failed to resume graph %s: %s", job_id, exc)
        raise HTTPException(status_code=500, detail=f"Failed to resume: {exc}")

    return ApproveResponse(job_id=job_id, engine="langgraph", status=new_status)


@router.get("/stock/{symbol}/quote", dependencies=[Depends(verify_key)], response_model=StockQuoteResponse)
async def stock_quote(symbol: str) -> StockQuoteResponse:
    """Real-time quote via yfinance (no agents)."""
    try:
        info = yf.Ticker(symbol).info
        return StockQuoteResponse(
            symbol=symbol.upper(),
            name=info.get("longName"),
            price=info.get("currentPrice", info.get("regularMarketPrice")),
            change_pct=info.get("regularMarketChangePercent"),
            market_cap=info.get("marketCap"),
            pe_ratio=info.get("trailingPE"),
            week_52_high=info.get("fiftyTwoWeekHigh"),
            week_52_low=info.get("fiftyTwoWeekLow"),
            sector=info.get("sector"),
            currency=info.get("currency", "USD"),
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not fetch data for {symbol}: {exc}")


@router.get("/stocks/popular", dependencies=[Depends(verify_key)])
async def popular_stocks() -> list[dict]:
    """Curated list of popular NSE + US stocks for the demo."""
    return POPULAR_STOCKS
