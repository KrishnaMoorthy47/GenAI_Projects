from __future__ import annotations

import json
import os
from typing import Any

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from pymongo import MongoClient
from tenacity import retry, stop_after_attempt, wait_exponential

from mealplanner.config import get_llm

SYSTEM_PROMPT = """
You are an expert at designing high protein, low glycemic, low carb dinners for couples.
You will be prompted to generate a weekly dinner plan.
You'll have access to recent meals. Factor these in so you aren't repetitive.
Bias towards meals that can be made in less than 30 minutes. Keep meal preparation simple.
There is no human in the loop, so don't prompt for additional input.
Format your final response as a JSON object with days of the week as keys.
"""


@tool
def get_adult_preferences() -> dict:
    """Get dietary preferences for adults."""
    return {
        "dietary_style": ["high protein", "low glycemic", "low carb"],
        "preferences": ["quick meals", "simple preparation", "under 30 minutes"],
    }


@tool
def get_recent_meals() -> list[dict[str, Any]]:
    """Get recent meals to avoid repetition."""
    client = MongoClient(os.getenv("MONGODB_URI", "mongodb://localhost:27017"))
    recent = list(client["meal_planner"]["weekly_meal_plans"].find().sort([("$natural", -1)]).limit(2))
    client.close()
    return [
        {day: meal.get("meal", "") for day, meal in plan["weeklyPlan"].items()}
        for plan in recent
        if "weeklyPlan" in plan
    ]


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8), reraise=True)
async def _run_agent(agent, inputs: dict) -> dict:
    return await agent.ainvoke(inputs)


async def generate_adult_meal_plan() -> dict:
    llm = get_llm(temperature=0.7)
    agent = create_react_agent(llm, [get_adult_preferences, get_recent_meals], prompt=SYSTEM_PROMPT)
    inputs = {"messages": [HumanMessage(content="Plan 7 high-protein, low-carb dinners for adults.")]}
    response = await _run_agent(agent, inputs)
    content = response["messages"][-1].content
    try:
        j_start, j_end = content.find("{"), content.rfind("}")
        if j_start != -1 and j_end != -1:
            return json.loads(content[j_start: j_end + 1])
    except Exception:
        pass
    return {"error": "Could not parse adult meal plan", "raw": content}
