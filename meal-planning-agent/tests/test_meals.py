from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient


MOCK_PLAN = {
    "weeklyPlan": {
        "Monday": {
            "meal": "Grilled Chicken",
            "childModification": "Cut into pieces",
            "adultModification": "Add spicy seasoning",
            "ingredients": ["chicken breast", "olive oil", "garlic"],
            "prepTime": "25 minutes",
            "recipe": "Grill chicken until cooked through.",
        }
    },
    "groceryList": {
        "Produce": ["garlic"],
        "Meat and Seafood": ["chicken breast"],
        "Condiments and Spices": ["olive oil"],
    },
}

MOCK_PREFS = {"likes": ["chicken", "pasta"], "dislikes": ["fish"], "hardRequirements": ["nut-free"]}


@pytest.mark.asyncio
async def test_health(app):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


@pytest.mark.asyncio
async def test_set_preferences(app, headers):
    with patch("mealplanner.services.mongodb.save_preferences", new_callable=AsyncMock, return_value=MOCK_PREFS):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            r = await client.post("/api/v1/preferences", json=MOCK_PREFS, headers=headers)
    assert r.status_code == 200
    assert r.json()["likes"] == ["chicken", "pasta"]


@pytest.mark.asyncio
async def test_get_preferences(app, headers):
    with patch("mealplanner.services.mongodb.get_preferences", new_callable=AsyncMock, return_value=MOCK_PREFS):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            r = await client.get("/api/v1/preferences", headers=headers)
    assert r.status_code == 200
    assert "hardRequirements" in r.json()


@pytest.mark.asyncio
async def test_generate_meal_plan(app, headers):
    with patch("mealplanner.api.meals.run_pipeline", new_callable=AsyncMock):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            r = await client.post("/api/v1/meal-plan/generate", headers=headers)
    assert r.status_code == 200
    data = r.json()
    assert "plan_id" in data
    assert data["status"] == "pending"


@pytest.mark.asyncio
async def test_plan_status_not_found(app, headers):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.get("/api/v1/meal-plan/nonexistent/status", headers=headers)
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_recent_plans(app, headers):
    with patch("mealplanner.services.mongodb.get_recent_meal_plans", new_callable=AsyncMock, return_value=[MOCK_PLAN]):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            r = await client.get("/api/v1/meal-plans/recent", headers=headers)
    assert r.status_code == 200
    assert isinstance(r.json(), list)


@pytest.mark.asyncio
async def test_unauthorized(app):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.get("/api/v1/preferences", headers={"x-api-key": "wrong"})
    assert r.status_code == 401
