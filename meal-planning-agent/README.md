# Meal Planning Agent with Streamlit

This project implements a multi-agent system for meal planning using Streamlit. The system helps families create balanced meal plans that accommodate different preferences and dietary requirements.

## Project Structure

```
meal-planning-agent/
├── agents/                    # Agent implementations
│   ├── child_agent/          # Agent for planning child-friendly meals
│   ├── adult_agent/          # Agent for planning adult meals
│   ├── shared_meal_agent/    # Agent for combining meal plans
│   └── format_output_agent/  # Agent for formatting final output
├── streamlit_app/            # Streamlit web application
└── README.md                 # Project documentation
```

## Features

- Multi-agent architecture with specialized agents for different meal planning tasks
- Streamlined design with Streamlit for the web interface and agent coordination
- Interactive UI for configuring preferences and viewing meal plans
- MongoDB for storing meal preferences and generated meal plans

## Prerequisites

- Python 3.11+
- MongoDB
- Anthropic API key

## Setup and Installation

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Set up environment variables in a `.env` file:
   ```
   MONGODB_URI=mongodb://localhost:27017
   ANTHROPIC_API_KEY=your_anthropic_api_key
   ```

3. Start the Streamlit application:
   ```
   cd streamlit_app
   streamlit run app.py
   ```

## How It Works

1. Users configure their meal preferences through the Streamlit interface
2. When a meal plan is requested, the child and adult meal planning agents process the request in parallel
3. The shared meal plan agent combines the outputs from both agents
4. The format output agent creates a structured response with a grocery list
5. The final meal plan is stored in MongoDB and displayed to the user in the Streamlit UI

## License

MIT
