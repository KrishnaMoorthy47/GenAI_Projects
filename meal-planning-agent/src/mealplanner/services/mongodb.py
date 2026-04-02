from __future__ import annotations

import datetime
import logging
from typing import Any

from motor.motor_asyncio import AsyncIOMotorClient

from mealplanner.config import get_settings

logger = logging.getLogger(__name__)

_client: AsyncIOMotorClient | None = None


def get_db():
    return _client["meal_planner"]


async def init_mongodb() -> None:
    global _client
    settings = get_settings()
    _client = AsyncIOMotorClient(settings.mongodb_uri)
    db = get_db()
    # Ensure collections exist
    names = await db.list_collection_names()
    if "meal_preferences" not in names:
        await db.create_collection("meal_preferences")
    if "weekly_meal_plans" not in names:
        await db.create_collection("weekly_meal_plans")
    logger.info("MongoDB connected and collections ready")


async def close_mongodb() -> None:
    global _client
    if _client:
        _client.close()
        logger.info("MongoDB connection closed")


async def save_preferences(likes: list, dislikes: list, hard_requirements: list) -> dict:
    db = get_db()
    preferences = {
        "likes": likes,
        "dislikes": dislikes,
        "hardRequirements": hard_requirements,
        "updated_at": datetime.datetime.utcnow(),
    }
    await db.meal_preferences.update_one({}, {"$set": preferences}, upsert=True)
    return {k: v for k, v in preferences.items() if k != "updated_at"}


async def get_preferences() -> dict:
    db = get_db()
    result = await db.meal_preferences.find_one({}, {"_id": 0})
    if not result:
        return {"likes": [], "dislikes": [], "hardRequirements": []}
    result.pop("updated_at", None)
    return result


async def save_meal_plan(plan: dict) -> str:
    db = get_db()
    plan = dict(plan)
    plan["created_at"] = datetime.datetime.utcnow()
    result = await db.weekly_meal_plans.insert_one(plan)
    return str(result.inserted_id)


async def get_recent_meal_plans(limit: int = 5) -> list[dict[str, Any]]:
    db = get_db()
    cursor = db.weekly_meal_plans.find({}, {"_id": 0}).sort("created_at", -1).limit(limit)
    plans = []
    async for doc in cursor:
        doc.pop("created_at", None)
        plans.append(doc)
    return plans
