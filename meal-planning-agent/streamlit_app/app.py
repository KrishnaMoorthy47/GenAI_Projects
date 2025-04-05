import streamlit as st
import os
import json
import datetime
from pymongo import MongoClient
import sys
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.child_agent.agent import generate_child_meal_plan
from agents.adult_agent.agent import generate_adult_meal_plan
from agents.shared_meal_agent.agent import generate_shared_meal_plan
from agents.format_output_agent.agent import format_output

@st.cache_resource
def get_mongo_client():
    mongo_uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
    return MongoClient(mongo_uri)

def init_db():
    client = get_mongo_client()
    db = client["meal_planner"]
    if "meal_preferences" not in db.list_collection_names():
        db.create_collection("meal_preferences")
    if "weekly_meal_plans" not in db.list_collection_names():
        db.create_collection("weekly_meal_plans")
    return db

def save_preferences(likes, dislikes, hard_requirements):
    db = init_db()
    preferences = {
        "likes": likes,
        "dislikes": dislikes,
        "hardRequirements": hard_requirements,
        "updated_at": datetime.datetime.now()
    }
    db.meal_preferences.update_one({}, {"$set": preferences}, upsert=True)
    return preferences

def get_preferences():
    db = init_db()
    preferences = db.meal_preferences.find_one({})
    if not preferences:
        return {"likes": [], "dislikes": [], "hardRequirements": []}
    return preferences

def save_meal_plan(meal_plan):
    db = init_db()
    meal_plan["created_at"] = datetime.datetime.now()
    result = db.weekly_meal_plans.insert_one(meal_plan)
    return result.inserted_id

def get_recent_meal_plans(limit=5):
    db = init_db()
    recent_plans = list(db.weekly_meal_plans.find().sort("created_at", -1).limit(limit))
    return recent_plans

def generate_meal_plan():
    with st.spinner("Generating meal plan..."):
        # Step 1: Generate child meal plan
        child_plan = generate_child_meal_plan()
        st.session_state.child_plan = child_plan
        
        # Step 2: Generate adult meal plan
        adult_plan = generate_adult_meal_plan()
        st.session_state.adult_plan = adult_plan
        
        # Step 3: Generate shared meal plan
        shared_plan = generate_shared_meal_plan(child_plan, adult_plan)
        st.session_state.shared_plan = shared_plan
        
        # Step 4: Format the output
        formatted_output = format_output(shared_plan)
        st.session_state.formatted_output = formatted_output
        
        # Step 5: Save to MongoDB
        save_meal_plan(formatted_output)
        
        return formatted_output

# Display meal plan
def display_meal_plan(meal_plan):
    st.subheader("Weekly Meal Plan")
    
    # Display meals for each day
    for day, meals in meal_plan["weeklyPlan"].items():
        st.write(f"### {day}")
        st.write(f"**Meal:** {meals['meal']}")
        
        # Display modifications if they exist
        if "childModification" in meals and meals["childModification"]:
            st.write(f"**Child Version:** {meals['childModification']}")
        if "adultModification" in meals and meals["adultModification"]:
            st.write(f"**Adult Version:** {meals['adultModification']}")
        
        # Display ingredients and recipe
        st.write("**Core Ingredients:**")
        for ingredient in meals["ingredients"]:
            st.write(f"- {ingredient}")
        
        st.write(f"**Prep Time:** {meals['prepTime']}")
        
        with st.expander("View Recipe"):
            st.write(meals["recipe"])
        
        st.divider()
    
    # Display grocery list
    st.subheader("Grocery List")
    categories = meal_plan["groceryList"]
    
    # Create a DataFrame for the grocery list
    grocery_data = []
    for category, items in categories.items():
        for item in items:
            grocery_data.append({"Category": category, "Item": item})
    
    if grocery_data:
        grocery_df = pd.DataFrame(grocery_data)
        st.dataframe(grocery_df, use_container_width=True)

def main():
    st.set_page_config(page_title="Family Meal Planner", page_icon="🍽️", layout="wide")
    st.title("Family Meal Planner")
    
    # Initialize session state
    if "preferences_saved" not in st.session_state:
        st.session_state.preferences_saved = False
    if "meal_plan_generated" not in st.session_state:
        st.session_state.meal_plan_generated = False
    if "formatted_output" not in st.session_state:
        st.session_state.formatted_output = None
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["Meal Preferences", "Generate Meal Plan", "View Recent Plans"])
    
    with tab1:
        st.header("Set Meal Preferences")
        
        # Get existing preferences
        existing_prefs = get_preferences()
        
        # Input for likes
        likes_default = ", ".join(existing_prefs.get("likes", []))
        likes = st.text_area("Foods your kids like (comma separated)", value=likes_default)
        
        # Input for dislikes
        dislikes_default = ", ".join(existing_prefs.get("dislikes", []))
        dislikes = st.text_area("Foods your kids dislike (comma separated)", value=dislikes_default)
        
        # Input for hard requirements
        hard_reqs_default = ", ".join(existing_prefs.get("hardRequirements", []))
        hard_requirements = st.text_area("Hard requirements (allergies, dietary restrictions, etc.)", value=hard_reqs_default)
        
        if st.button("Save Preferences"):
            likes_list = [item.strip() for item in likes.split(",") if item.strip()]
            dislikes_list = [item.strip() for item in dislikes.split(",") if item.strip()]
            hard_reqs_list = [item.strip() for item in hard_requirements.split(",") if item.strip()]
            
            save_preferences(likes_list, dislikes_list, hard_reqs_list)
            st.session_state.preferences_saved = True
            st.success("Preferences saved successfully!")
    
    with tab2:
        st.header("Generate Weekly Meal Plan")
        
        if st.button("Generate New Meal Plan"):
            meal_plan = generate_meal_plan()
            st.session_state.meal_plan_generated = True
        
        if st.session_state.meal_plan_generated and st.session_state.formatted_output:
            display_meal_plan(st.session_state.formatted_output)
    
    with tab3:
        st.header("Recent Meal Plans")
        recent_plans = get_recent_meal_plans()
        
        if not recent_plans:
            st.info("No meal plans found. Generate a new meal plan to see it here.")
        else:
            for i, plan in enumerate(recent_plans):
                created_at = plan.get("created_at", datetime.datetime.now()).strftime("%Y-%m-%d %H:%M")
                with st.expander(f"Meal Plan {i+1} - Created on {created_at}"):
                    display_meal_plan(plan)

if __name__ == "__main__":
    main()
