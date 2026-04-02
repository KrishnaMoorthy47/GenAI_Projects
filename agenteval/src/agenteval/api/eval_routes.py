"""Eval run management endpoints."""

import asyncio
import logging
import uuid

from fastapi import APIRouter, HTTPException, Request

from agenteval.eval.runner import EvalRunner
from agenteval.models.request import EvalRunRequest
from agenteval.models.response import EvalReport

_REPORT_STORE_MAX_SIZE = 100


def _evict_oldest_reports(store: dict) -> None:
    """Keep report_store bounded by evicting the oldest completed entries."""
    if len(store) <= _REPORT_STORE_MAX_SIZE:
        return
    terminal = [k for k, v in store.items() if getattr(v, "status", "") in ("completed", "failed")]
    for key in terminal[: len(store) - _REPORT_STORE_MAX_SIZE]:
        del store[key]
    if len(store) > _REPORT_STORE_MAX_SIZE:
        excess = list(store.keys())[: len(store) - _REPORT_STORE_MAX_SIZE]
        for key in excess:
            del store[key]

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/eval", tags=["eval"])


@router.post("/run-eval", status_code=202)
async def run_eval(body: EvalRunRequest, request: Request):
    """Start an async eval run. Returns run_id immediately."""
    report_store: dict = request.app.state.report_store
    _evict_oldest_reports(report_store)

    run_id = str(uuid.uuid4())
    report_store[run_id] = EvalReport(run_id=run_id, status="pending")

    runner = EvalRunner(
        report_store=report_store,
        dataset_path=body.dataset_path,
    )
    asyncio.create_task(
        runner.run(
            run_id=run_id,
            push_langsmith=body.push_to_langsmith,
            experiment_prefix=body.experiment_prefix or "sql-agent",
        )
    )
    return {"run_id": run_id, "status": "pending"}


@router.get("/{run_id}/status")
async def get_status(run_id: str, request: Request):
    """Poll eval run status."""
    report_store: dict = request.app.state.report_store
    report = report_store.get(run_id)
    if not report:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")
    return {
        "run_id": run_id,
        "status": report.status,
        "total_cases": report.total_cases,
        "completed_cases": report.completed_cases,
    }


@router.get("/{run_id}/report", response_model=EvalReport)
async def get_report(run_id: str, request: Request):
    """Get the full EvalReport for a completed run."""
    report_store: dict = request.app.state.report_store
    report = report_store.get(run_id)
    if not report:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")
    if report.status not in ("completed", "failed"):
        raise HTTPException(status_code=202, detail=f"Run still {report.status}")
    return report
