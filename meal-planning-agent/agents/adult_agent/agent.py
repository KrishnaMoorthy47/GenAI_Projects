import os
import json
from typing import Dict, List, Any, Optional, Sequence, TypedDict, Annotated
from pymongo import MongoClient
from langchain_core.prompts import ChatPromptTemplate
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field

SYSTEM_PROMPT = """
You are an expert at designing high protein, low glycemic, low carb dinners for couples.
You will be prompted to generate a weekly dinner plan.
You'll have access to recent meals. Factor these in so you aren't repetitive.
Bias towards meals that can be made in less than 30 minutes. Keep meal preparation simple.
There is no human in the loop, so don't prompt for additional input.
"""

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], "The messages in the conversation"]

# Tool to get recent meals from MongoDB
@tool
def get_recent_meals() -> List[Dict[str, Any]]:
    """Use this to get recent meals."""
    client = MongoClient(os.getenv("MONGODB_URI", "mongodb://localhost:27017"))
    
    db = client['meal_planner']
    collection = db['weekly_meal_plans']
    
    # Query to get the last two entries
    recent_meals = list(collection.find().sort([("$natural", -1)]).limit(2))
    
    formatted_meals = []
    for plan in recent_meals:
        if "weeklyPlan" in plan:
            meal_names = {}
            for day, meal in plan["weeklyPlan"].items():
                meal_names[day] = meal.get("meal", "")
            formatted_meals.append(meal_names)
    
    return formatted_meals

# Tool to get the number of meals to plan
@tool
def get_meal_count() -> int:
    """Use this to get the number of meals to plan for the week."""
    return 7  # Default to 7 days (full week)

# Tool to get dietary preferences for adults
@tool
def get_adult_preferences() -> Dict[str, List[str]]:
    """Use this to get dietary preferences for adults."""
    return {
        "dietary_style": ["high protein", "low glycemic", "low carb"],
        "preferences": ["quick meals", "simple preparation"]
    }

def build_adult_meal_planning_agent():
    llm = ChatAnthropic(model="claude-3-sonnet-20240229", temperature=0.7)
    
    tools = [get_recent_meals, get_meal_count, get_adult_preferences]
    
    agent = create_react_agent(llm, tools, system_prompt=SYSTEM_PROMPT)
    
    return agent

def generate_adult_meal_plan(request_id="default"):
    agent = build_adult_meal_planning_agent()
    
    meal_count = get_meal_count()
    inputs = {"messages": [HumanMessage(content=f"Plan {meal_count} high-protein, low-carb dinners for adults.")]}
    
    response = agent.invoke(inputs)
    
    last_message = response["messages"][-1]
    content = last_message.content
    
    # Extract JSON from the response
    try:
        json_start = content.find('{')
        json_end = content.rfind('}')
        
        if json_start != -1 and json_end != -1:
            json_str = content[json_start:json_end+1]
            meal_plan = json.loads(json_str)
        else:
            meal_plan = {"error": "Could not parse JSON", "raw_response": content}
    except Exception as e:
        meal_plan = {"error": str(e), "raw_response": content}
    
    return meal_plan
