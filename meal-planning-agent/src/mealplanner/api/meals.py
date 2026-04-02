from __future__ import annotations

import asyncio
import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Security
from fastapi.security.api_key import APIKeyHeader

from mealplanner.config import get_settings
from mealplanner.models.request import PreferencesRequest
from mealplanner.models.response import PlanStartResponse, PlanStatusResponse
from mealplanner.services import mongodb
from mealplanner.services.pipeline import (
    PlanStatus,
    create_plan_entry,
    get_plan_store,
    run_pipeline,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1")

_api_key_header = APIKeyHeader(name="x-api-key", auto_error=True)


def verify_key(key: str = Security(_api_key_header)) -> str:
    if key != get_settings().api_key:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return key


# ── Preferences ──────────────────────────────────────────────────────────────


@router.post("/preferences", dependencies=[Depends(verify_key)])
async def set_preferences(body: PreferencesRequest) -> dict:
    """Save meal preferences (likes, dislikes, dietary restrictions)."""
    return await mongodb.save_preferences(body.likes, body.dislikes, body.hard_requirements)


@router.get("/preferences", dependencies=[Depends(verify_key)])
async def get_preferences() -> dict:
    """Get current meal preferences."""
    return await mongodb.get_preferences()


# ── Meal Plan Generation ──────────────────────────────────────────────────────


@router.post("/meal-plan/generate", dependencies=[Depends(verify_key)], response_model=PlanStartResponse)
async def generate_meal_plan() -> PlanStartResponse:
    """
    Kick off a 4-agent meal plan generation pipeline.
    Returns immediately with a plan_id. Poll /meal-plan/{plan_id}/status for progress.
    """
    plan_id = create_plan_entry()
    asyncio.create_task(run_pipeline(plan_id))
    return PlanStartResponse(plan_id=plan_id, status=PlanStatus.PENDING)


@router.get("/meal-plan/{plan_id}/status", dependencies=[Depends(verify_key)], response_model=PlanStatusResponse)
async def get_plan_status(plan_id: str) -> PlanStatusResponse:
    """Poll the status of a running meal plan generation."""
    store = get_plan_store()
    if plan_id not in store:
        raise HTTPException(status_code=404, detail="Plan not found")
    entry = store[plan_id]
    return PlanStatusResponse(
        plan_id=plan_id,
        status=entry["status"],
        step=entry["step"],
        error=entry.get("error"),
    )


@router.get("/meal-plan/{plan_id}", dependencies=[Depends(verify_key)])
async def get_plan_result(plan_id: str) -> dict[str, Any]:
    """Get the completed meal plan (only available when status=completed)."""
    store = get_plan_store()
    if plan_id not in store:
        raise HTTPException(status_code=404, detail="Plan not found")
    entry = store[plan_id]
    if entry["status"] != PlanStatus.COMPLETED:
        raise HTTPException(status_code=409, detail=f"Plan not ready — status: {entry['status']}")
    return entry["result"]


# ── Recent Plans ──────────────────────────────────────────────────────────────


@router.get("/meal-plans/recent", dependencies=[Depends(verify_key)])
async def recent_plans(limit: int = 5) -> list[dict[str, Any]]:
    """Get the last N meal plans from MongoDB."""
    return await mongodb.get_recent_meal_plans(limit)
