from __future__ import annotations

import json

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from tenacity import retry, stop_after_attempt, wait_exponential

from mealplanner.config import get_llm

SYSTEM_PROMPT = """
You are a meal planning assistant responsible for formatting the final output of a family meal plan.
Your task is to take a combined meal plan and create a structured response with a comprehensive grocery list.

The grocery list should be organized by categories:
- Produce
- Meat and Seafood
- Dairy and Eggs
- Grains and Bread
- Canned and Jarred Goods
- Frozen Foods
- Condiments and Spices
- Other

Format your response as a JSON object exactly like this:
{
  "weeklyPlan": {
    "Monday": {
      "meal": "Meal Name",
      "childModification": "...",
      "adultModification": "...",
      "ingredients": ["ingredient1", ...],
      "prepTime": "25 minutes",
      "recipe": "Brief recipe instructions"
    }
  },
  "groceryList": {
    "Produce": ["item1", ...],
    "Meat and Seafood": [...]
  }
}
"""


@tool
def get_grocery_categories() -> list[str]:
    """Get the list of grocery categories."""
    return ["Produce", "Meat and Seafood", "Dairy and Eggs", "Grains and Bread",
            "Canned and Jarred Goods", "Frozen Foods", "Condiments and Spices", "Other"]


@tool
def get_days_of_week() -> list[str]:
    """Get the days of the week."""
    return ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8), reraise=True)
async def _run_agent(agent, inputs: dict) -> dict:
    return await agent.ainvoke(inputs)


async def format_output(shared_meal_plan: dict) -> dict:
    llm = get_llm(temperature=0.5)
    agent = create_react_agent(llm, [get_grocery_categories, get_days_of_week], prompt=SYSTEM_PROMPT)

    prompt = f"""
Please format this meal plan and create a comprehensive grocery list organized by category:

{json.dumps(shared_meal_plan, indent=2)}

Return the complete structured JSON with weeklyPlan and groceryList.
"""
    response = await _run_agent(agent, {"messages": [HumanMessage(content=prompt)]})
    content = response["messages"][-1].content
    try:
        j_start, j_end = content.find("{"), content.rfind("}")
        if j_start != -1 and j_end != -1:
            return json.loads(content[j_start: j_end + 1])
    except Exception:
        pass
    return {"error": "Could not parse formatted output", "raw": content}
