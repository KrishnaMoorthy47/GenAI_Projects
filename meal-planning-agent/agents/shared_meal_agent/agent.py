import os
import json
from typing import Dict, List, Any, Optional, Sequence, TypedDict, Annotated
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field

# System prompt for shared meal planning
SYSTEM_PROMPT = """
You are an expert at combining meal plans to create a cohesive family dinner plan.
You will be given two meal plans: one for children and one for adults.
Your task is to create a shared meal plan that accommodates both preferences.
When possible, suggest modifications to make a single meal work for everyone.
When not possible, provide separate meal options but try to minimize extra cooking.
You should aim to create meals that are nutritious, tasty, and have good variety.
There is no human in the loop, so don't prompt for additional input.
"""

# Define the state for the agent
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], "The messages in the conversation"]

# Tool to combine meal plans
@tool
def combine_meal_plans(child_plan: Dict[str, Any], adult_plan: Dict[str, Any]) -> Dict[str, Any]:
    """
    Combine the child and adult meal plans into a cohesive family meal plan.
    This tool takes the child meal plan and adult meal plan as input and returns a combined plan.
    """
    # This is just a placeholder - the actual combination will be done by the LLM
    return {"combined": "placeholder"}

# Tool to get days of the week
@tool
def get_days_of_week() -> List[str]:
    """Get the days of the week for meal planning."""
    return ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

# Tool to evaluate meal plan nutrition
@tool
def evaluate_nutrition(meal_plan: Dict[str, Any]) -> Dict[str, str]:
    """Evaluate the nutritional balance of a meal plan."""
    # This is just a placeholder - the actual evaluation will be done by the LLM
    return {"evaluation": "placeholder"}

# Function to generate a shared meal plan with reflection
def generate_shared_meal_plan(child_meal_plan, adult_meal_plan):
    # Initialize the LLM
    llm = ChatAnthropic(model="claude-3-sonnet-20240229", temperature=0.7)
    
    # Format the meal plans for the prompt
    child_plan_str = json.dumps(child_meal_plan, indent=2)
    adult_plan_str = json.dumps(adult_meal_plan, indent=2)
    
    # Create the prompt with both meal plans
    prompt = f"""
    I need to create a shared family meal plan that works for both children and adults.
    
    Here is the meal plan for children:
    {child_plan_str}
    
    Here is the meal plan for adults (high protein, low carb):
    {adult_plan_str}
    
    Please combine these into a cohesive family meal plan. For each day:
    1. If possible, suggest a single meal that works for everyone (with minor modifications if needed)
    2. If not possible, provide separate options but try to minimize extra cooking
    3. Include any modifications needed to make meals work for different family members
    
    Format your response as a structured JSON object with days of the week as keys.
    Each day should include information about whether it's a shared meal or separate meals.
    
    After creating the initial plan, please review it for:
    - Nutritional balance
    - Variety of foods and flavors
    - Family-friendliness
    - Ease of preparation
    
    Then provide an improved version based on your review.
    """
    
    # Create the input message
    inputs = {"messages": [HumanMessage(content=prompt)]}
    
    # Create the ReAct agent
    tools = [combine_meal_plans, get_days_of_week, evaluate_nutrition]
    agent = create_react_agent(llm, tools, system_prompt=SYSTEM_PROMPT)
    
    # Invoke the agent
    response = agent.invoke(inputs)
    
    # Extract the last message content
    last_message = response["messages"][-1]
    content = last_message.content
    
    # Extract JSON from the response
    try:
        # Look for JSON in the response
        json_start = content.find('{')
        json_end = content.rfind('}')
        
        if json_start != -1 and json_end != -1:
            json_str = content[json_start:json_end+1]
            shared_meal_plan = json.loads(json_str)
        else:
            # If no JSON found, try to structure the response
            shared_meal_plan = {"error": "Could not parse JSON", "raw_response": content}
    except Exception as e:
        shared_meal_plan = {"error": str(e), "raw_response": content}
    
    return shared_meal_plan

# For testing
if __name__ == "__main__":
    # Example meal plans
    child_plan = {
        "Monday": {"meal": "Chicken Nuggets", "ingredients": ["chicken", "breadcrumbs"], "prepTime": "20 minutes", "recipe": "Bake nuggets until golden."}
    }
    adult_plan = {
        "Monday": {"meal": "Grilled Chicken Salad", "ingredients": ["chicken", "lettuce", "tomatoes"], "prepTime": "15 minutes", "recipe": "Grill chicken and toss with vegetables."}
    }
    
    shared_plan = generate_shared_meal_plan(child_plan, adult_plan)
    print(json.dumps(shared_plan, indent=2))
