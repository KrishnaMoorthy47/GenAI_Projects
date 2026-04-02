# AgentEval — Architecture

LangGraph ReAct SQL agent paired with a 4-scorer evaluation framework. The agent answers natural language questions over a SQLite database using list_tables → get_schema → run_query tool calls. The eval harness runs 20 curated test cases and scores each on task success, tool accuracy, trajectory efficiency, and hallucination. Results are tracked in LangSmith.

## Flow

```
POST /agent/query              POST /eval/run-eval
      │                               │  asyncio.create_task()
      ▼                               ▼
  SQL Agent (ReAct loop)         EvalRunner
      │                               │
  agent_node (LLM)              run_sql_agent × 20 cases
      │ tool calls?                   │
      ├─ Yes ──► tool_node            ▼  score each case:
      │          │              task_success   (LLM-as-judge)
      │◄─────────┘              tool_accuracy  (deterministic)
      │                         trajectory     (deterministic)
      └─ No ──► END             hallucination  (LLM-as-judge)
                                     │
                               EvalReport → LangSmith
```

## API

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| GET | `/health` | No | Health check |
| POST | `/agent/query` | Yes | Run a single SQL agent query |
| POST | `/eval/run-eval` | Yes | Start a full evaluation run |
| GET | `/eval/{id}/status` | Yes | Poll evaluation progress |
| GET | `/eval/{id}/report` | Yes | Retrieve completed eval report |

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `API_KEY` | required | `x-api-key` header value |
| `LLM_PROVIDER` | `openai` | `openai` \| `azure_openai` |
| `OPENAI_API_KEY` | — | OpenAI API key |
| `LANGSMITH_API_KEY` | — | LangSmith tracking (optional) |
| `LANGSMITH_PROJECT` | `agenteval` | LangSmith project name |
| `MAX_AGENT_ITERATIONS` | `8` | ReAct loop iteration cap |
