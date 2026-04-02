# Meal Planning Agent — Architecture

AI-powered family meal planner using a 4-agent LangGraph pipeline. Preferences (likes, dislikes, dietary restrictions) are stored in MongoDB Atlas. On request, four specialised agents generate adult dinners, child-friendly dinners, a shared family meal, and a formatted weekly plan with a categorised grocery list.

## Flow

```
POST /api/v1/meal-plan/generate
      │
      ▼  asyncio.create_task()
┌─────────────────────────────────┐
│  LangGraph Pipeline             │
│                                 │
│  adult_agent                    │
│       → child_agent             │
│       → shared_meal_agent       │
│       → format_output_agent     │
│       → save to MongoDB         │
└─────────────────────────────────┘
      │
  MongoDB Atlas  (preferences + weekly plans)
```

## API

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| GET | `/health` | No | Service health check |
| POST | `/api/v1/preferences` | Yes | Save meal preferences |
| GET | `/api/v1/preferences` | Yes | Get saved preferences |
| POST | `/api/v1/meal-plan/generate` | Yes | Kick off plan generation |
| GET | `/api/v1/meal-plan/{id}/status` | Yes | Poll generation progress |
| GET | `/api/v1/meal-plan/{id}` | Yes | Get completed meal plan |
| GET | `/api/v1/meal-plans/recent` | Yes | Get last N saved plans |

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `API_KEY` | required | `x-api-key` header value |
| `LLM_PROVIDER` | `groq` | `groq` \| `openai` \| `azure_openai` |
| `GROQ_API_KEY` | — | Groq API key |
| `OPENAI_API_KEY` | — | OpenAI API key |
| `MONGODB_URI` | required | MongoDB connection string |
| `PORT` | `8006` | Server port |
