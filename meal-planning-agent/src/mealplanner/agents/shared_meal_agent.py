from __future__ import annotations

import json

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from tenacity import retry, stop_after_attempt, wait_exponential

from mealplanner.config import get_llm

SYSTEM_PROMPT = """
You are an expert at combining meal plans to create a cohesive family dinner plan.
You will be given two meal plans: one for children and one for adults.
Your task is to create a shared meal plan that accommodates both preferences.
When possible, suggest modifications to make a single meal work for everyone.
When not possible, provide separate meal options but try to minimize extra cooking.
You should aim to create meals that are nutritious, tasty, and have good variety.
There is no human in the loop, so don't prompt for additional input.
Format your response as a JSON object with days of the week as keys.
"""


@tool
def get_days_of_week() -> list[str]:
    """Get the days of the week for meal planning."""
    return ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8), reraise=True)
async def _run_agent(agent, inputs: dict) -> dict:
    return await agent.ainvoke(inputs)


async def generate_shared_meal_plan(child_plan: dict, adult_plan: dict) -> dict:
    llm = get_llm(temperature=0.7)
    agent = create_react_agent(llm, [get_days_of_week], prompt=SYSTEM_PROMPT)

    prompt = f"""
I need to create a shared family meal plan that works for both children and adults.

Children's meal plan:
{json.dumps(child_plan, indent=2)}

Adults' meal plan (high protein, low carb):
{json.dumps(adult_plan, indent=2)}

Combine these into a cohesive family meal plan. For each day:
1. Suggest a single meal that works for everyone (with minor modifications if needed)
2. If not possible, provide separate options but minimise extra cooking
3. Include any modifications needed for different family members

Format as a JSON object with days as keys and meal details as values.
"""
    response = await _run_agent(agent, {"messages": [HumanMessage(content=prompt)]})
    content = response["messages"][-1].content
    try:
        j_start, j_end = content.find("{"), content.rfind("}")
        if j_start != -1 and j_end != -1:
            return json.loads(content[j_start: j_end + 1])
    except Exception:
        pass
    return {"error": "Could not parse shared meal plan", "raw": content}
