import os
import json
from typing import Dict, List, Any, Optional, Sequence, TypedDict, Annotated
from langchain_core.prompts import ChatPromptTemplate
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field

SYSTEM_PROMPT = """
You are a meal planning assistant responsible for formatting the final output of a family meal plan.
Your task is to take a combined meal plan and create a structured response with a comprehensive grocery list.

The grocery list should be organized by categories such as:
- Produce
- Meat and Seafood
- Dairy and Eggs
- Grains and Bread
- Canned and Jarred Goods
- Frozen Foods
- Condiments and Spices
- Snacks
- Other

For each meal in the plan, ensure that all ingredients are included in the grocery list under the appropriate category.
Format your response as a structured JSON object with the following format:
{
  "weeklyPlan": {
    "Monday": {
      "meal": "Meal Name",
      "childModification": "Any modifications for children (if applicable)",
      "adultModification": "Any modifications for adults (if applicable)",
      "ingredients": ["ingredient1", "ingredient2", ...],
      "prepTime": "Preparation time",
      "recipe": "Brief recipe instructions"
    },
    ...
  },
  "groceryList": {
    "Produce": ["item1", "item2", ...],
    "Meat and Seafood": [...],
    ...
  }
}
"""

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], "The messages in the conversation"]

@tool
def extract_ingredients(meal_plan: Dict[str, Any]) -> List[str]:
    """Extract all ingredients from the meal plan."""
    all_ingredients = set()
    for day, meal_data in meal_plan.items():
        if isinstance(meal_data, dict) and "ingredients" in meal_data:
            all_ingredients.update(meal_data["ingredients"])
    return list(all_ingredients)

# Tool to categorize ingredients
@tool
def categorize_ingredient(ingredient: str) -> str:
    """Categorize a single ingredient into one of the grocery categories."""
    return "Unknown"

# Tool to get grocery categories
@tool
def get_grocery_categories() -> List[str]:
    """Get the list of grocery categories for organizing the grocery list."""
    return [
        "Produce",
        "Meat and Seafood",
        "Dairy and Eggs",
        "Grains and Bread",
        "Canned and Jarred Goods",
        "Frozen Foods",
        "Condiments and Spices",
        "Snacks",
        "Other"
    ]

# Tool to get days of the week
@tool
def get_days_of_week() -> List[str]:
    """Get the days of the week for meal planning."""
    return ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

# Function to format the output
def format_output(shared_meal_plan):
    llm = ChatAnthropic(model="claude-3-sonnet-20240229", temperature=0.5)
    tools = [extract_ingredients, categorize_ingredient, get_grocery_categories, get_days_of_week]
    agent = create_react_agent(llm, tools, system_prompt=SYSTEM_PROMPT)
    
    # Format the shared meal plan as a JSON string
    shared_plan_str = json.dumps(shared_meal_plan, indent=2)
    
    prompt = f"""
    Please format this meal plan and create a comprehensive grocery list:
    
    {shared_plan_str}
    
    Organize all ingredients into appropriate categories for the grocery list.
    """
    
    inputs = {"messages": [HumanMessage(content=prompt)]}
    
    response = agent.invoke(inputs)
    
    last_message = response["messages"][-1]
    content = last_message.content
    
    try:
        json_start = content.find('{')
        json_end = content.rfind('}')
        
        if json_start != -1 and json_end != -1:
            json_str = content[json_start:json_end+1]
            formatted_output = json.loads(json_str)
        else:
            formatted_output = {"error": "Could not parse JSON", "raw_response": content}
    except Exception as e:
        formatted_output = {"error": str(e), "raw_response": content}
    
    return formatted_output

if __name__ == "__main__":
    shared_plan = {
        "Monday": {
            "meal": "Grilled Chicken with Vegetables",
            "childModification": "Chicken cut into nugget-sized pieces",
            "adultModification": "Add spicy seasoning",
            "ingredients": ["chicken breast", "broccoli", "carrots", "olive oil", "garlic"],
            "prepTime": "25 minutes",
            "recipe": "Grill chicken and steam vegetables."
        }
    }
    
    formatted_output = format_output(shared_plan)
    print(json.dumps(formatted_output, indent=2))
