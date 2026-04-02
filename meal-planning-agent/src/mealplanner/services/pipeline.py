from __future__ import annotations

import asyncio
import logging
import uuid
from enum import Enum
from typing import Any

from mealplanner.agents.adult_agent import generate_adult_meal_plan
from mealplanner.agents.child_agent import generate_child_meal_plan
from mealplanner.agents.format_output_agent import format_output
from mealplanner.agents.shared_meal_agent import generate_shared_meal_plan
from mealplanner.services.mongodb import save_meal_plan

logger = logging.getLogger(__name__)


class PlanStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


# In-memory store — keyed by plan_id
_plan_store: dict[str, dict[str, Any]] = {}


def get_plan_store() -> dict[str, dict[str, Any]]:
    return _plan_store


def create_plan_entry() -> str:
    plan_id = str(uuid.uuid4())
    _plan_store[plan_id] = {"status": PlanStatus.PENDING, "step": "queued", "result": None, "error": None}
    return plan_id


async def run_pipeline(plan_id: str) -> None:
    store = _plan_store[plan_id]
    try:
        store["status"] = PlanStatus.RUNNING

        logger.info("[%s] Step 1/4 — child meal agent", plan_id)
        store["step"] = "child_agent"
        child_plan = await generate_child_meal_plan()

        logger.info("[%s] Step 2/4 — adult meal agent", plan_id)
        store["step"] = "adult_agent"
        adult_plan = await generate_adult_meal_plan()

        logger.info("[%s] Step 3/4 — shared meal agent", plan_id)
        store["step"] = "shared_meal_agent"
        shared_plan = await generate_shared_meal_plan(child_plan, adult_plan)

        logger.info("[%s] Step 4/4 — format output agent", plan_id)
        store["step"] = "format_output_agent"
        final_plan = await format_output(shared_plan)

        logger.info("[%s] Saving to MongoDB", plan_id)
        store["step"] = "saving"
        await save_meal_plan(final_plan)

        store["status"] = PlanStatus.COMPLETED
        store["step"] = "done"
        store["result"] = final_plan
        logger.info("[%s] Pipeline complete", plan_id)

    except Exception as exc:
        logger.exception("[%s] Pipeline failed: %s", plan_id, exc)
        store["status"] = PlanStatus.FAILED
        store["error"] = str(exc)
