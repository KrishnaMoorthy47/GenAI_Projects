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
You are an expert at designing nutritious meals that toddlers love.
You will be prompted to generate a weekly dinner plan.
You'll have access to meal preferences. Use these as inspiration to come up with meals but you don't have to explicitly use these items.
You'll have access to recent meals. Factor these in so you aren't repetitive.
You must take into account any hard requirements about meals.
Bias towards meals that can be made in less than 30 minutes.
Keep meal preparation simple. There is no human in the loop, so don't prompt for additional input.
"""

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], "The messages in the conversation"]

@tool
def get_kid_preferences() -> Dict[str, List[str]]:
    """Use this to get the likes and dislikes for the kids preferences."""
    client = MongoClient(os.getenv("MONGODB_URI", "mongodb://localhost:27017"))
    
    db = client['meal_planner']
    collection = db['meal_preferences']
    projection = {"likes": 1, "dislikes": 1, "_id": 0}
    result = collection.find_one({}, projection)
    return result or {"likes": [], "dislikes": []}

@tool
def get_hard_requirements() -> List[str]:
    """Use this to get the hard requirements for recommending a meal. These must be enforced."""
    client = MongoClient(os.getenv("MONGODB_URI", "mongodb://localhost:27017"))
    
    db = client['meal_planner']
    collection = db['meal_preferences']
    projection = {"hardRequirements": 1, "_id": 0}
    result = collection.find_one({}, projection)
    return result.get("hardRequirements", []) if result else []

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

@tool
def get_meal_count() -> int:
    """Use this to get the number of meals to plan for the week."""
    return 7  # Default to 7 days (full week)

def build_child_meal_planning_agent():
    llm = ChatAnthropic(model="claude-3-sonnet-20240229", temperature=0.7)
    tools = [get_kid_preferences, get_hard_requirements, get_recent_meals, get_meal_count]
    agent = create_react_agent(llm, tools, system_prompt=SYSTEM_PROMPT)
    
    return agent

def generate_child_meal_plan(request_id="default"):
    agent = build_child_meal_planning_agent()
    
    meal_count = get_meal_count()
    inputs = {"messages": [HumanMessage(content=f"Plan {meal_count} dinners for my children.")]}
    
    response = agent.invoke(inputs)
    
    last_message = response["messages"][-1]
    content = last_message.content
    
    try:
        # Look for JSON in the response
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

if __name__ == "__main__":
    meal_plan = generate_child_meal_plan()
    print(json.dumps(meal_plan, indent=2))
