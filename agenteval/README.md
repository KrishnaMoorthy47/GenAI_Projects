# AgentEval — AI Agent Evaluation Framework

A portfolio project demonstrating **LLM evaluation engineering**: measuring a LangGraph SQL agent on 4 dimensions — task success, tool accuracy, trajectory efficiency, and hallucination.

Most engineers build agents. This project *measures* them.

---

## What's Inside

| Component | Description |
|-----------|-------------|
| **SQL Agent** | LangGraph ReAct agent that answers natural language questions over a Tamil songs database (SQLite) |
| **Eval Framework** | Runs 20 test cases through the agent, scores each on 4 dimensions, aggregates results |
| **FastAPI** | REST API to trigger eval runs, poll status, and get reports; plus direct agent query endpoint |
| **LangSmith** | Experiment tracking: dataset push + per-run tracing |

---

## Architecture

```
POST /agent/query          → run_sql_agent() → LangGraph graph → SQLite tools
POST /eval/run-eval        → asyncio.create_task(EvalRunner.run())
GET  /eval/{id}/status     → poll EvalReport.status
GET  /eval/{id}/report     → full EvalReport with per-case scores
```

**SQL Agent Graph:**
```
START → agent_node (ReAct) ⟶ [has tool calls?] → tool_node → agent_node
                            ↘ [no tool calls]  → END
```

**4 Scorers:**

| Scorer | Type | Logic |
|--------|------|-------|
| `task_success` | LLM-as-judge | GPT-4o: does answer match expected? (0–1) |
| `tool_accuracy` | Deterministic | Recall: `|expected ∩ called| / |expected|` |
| `trajectory_efficiency` | Deterministic | `1.0 - 0.1 × extra_steps` |
| `hallucination` | LLM-as-judge | GPT-4o: is answer grounded in SQL result? (0–1) |

**Overall score:** weighted average (40% task, 20% tool, 20% trajectory, 20% hallucination)

---

## Quickstart

```bash
# 1. Install dependencies
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env
# Edit .env — set OPENAI_API_KEY, LANGSMITH_API_KEY, API_KEY

# 3. Run server
uv run uvicorn src.agenteval.main:app --port 8001 --reload

# 4. Ask the agent a question
curl -X POST http://localhost:8001/agent/query \
  -H "x-api-key: dev-secret" \
  -H "Content-Type: application/json" \
  -d '{"question": "How many songs are in the Melody genre?"}'

# 5. Run full evaluation
curl -X POST http://localhost:8001/eval/run-eval \
  -H "x-api-key: dev-secret" \
  -H "Content-Type: application/json" \
  -d '{"push_to_langsmith": true}'
# → {"run_id": "...", "status": "pending"}

# 6. Poll status
curl http://localhost:8001/eval/{run_id}/status -H "x-api-key: dev-secret"

# 7. Get report
curl http://localhost:8001/eval/{run_id}/report -H "x-api-key: dev-secret"

# 8. Run tests
uv run pytest tests/ -v
```

---

## Project Structure

```
agenteval/
├── src/agenteval/
│   ├── main.py              # FastAPI app (lifespan, auth, routers)
│   ├── config.py            # Settings (pydantic-settings) + get_llm()
│   ├── agent/
│   │   ├── state.py         # SQLAgentState TypedDict
│   │   ├── graph.py         # build_sql_agent() / run_sql_agent()
│   │   ├── nodes.py         # agent_node, tool_node
│   │   └── tools.py         # list_tables, get_schema, run_query (@tool)
│   ├── eval/
│   │   ├── runner.py        # EvalRunner orchestrator
│   │   ├── dataset.py       # load_cases(), push_to_langsmith()
│   │   ├── reporter.py      # aggregate_results() → EvalReport
│   │   └── scorers/
│   │       ├── task_success.py
│   │       ├── tool_accuracy.py
│   │       ├── trajectory.py
│   │       └── hallucination.py
│   ├── api/
│   │   ├── health.py        # GET /health
│   │   ├── agent_routes.py  # POST /agent/query
│   │   └── eval_routes.py   # POST /run-eval, GET /{id}/status|report
│   └── models/
│       ├── request.py       # EvalRunRequest, AgentQueryRequest
│       └── response.py      # EvalReport, CaseResult, ScoreCard
├── data/tamil_songs.db      # SQLite Tamil songs DB (8 tables, 92 songs)
├── datasets/eval_cases.json # 20 test cases (8 easy, 8 medium, 4 hard)
└── tests/                   # 42 tests (all mocked, no OpenAI calls)
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `API_KEY` | `dev-secret` | `x-api-key` header auth |
| `LLM_PROVIDER` | `openai` | `openai` or `azure_openai` |
| `OPENAI_API_KEY` | — | OpenAI API key |
| `OPENAI_MODEL` | `gpt-4o` | Model for agent + judges |
| `LANGSMITH_API_KEY` | — | LangSmith API key |
| `LANGSMITH_PROJECT` | `agenteval` | LangSmith project name |
| `LANGSMITH_TRACING` | `false` | Enable LangSmith tracing |
| `PORT` | `8001` | Server port |
| `DB_PATH` | `data/tamil_songs.db` | SQLite DB path |

---

## Test Dataset

20 questions about the Tamil songs database:
- **8 easy**: single-table counts/lookups (`How many singers?`)
- **8 medium**: joins + aggregations (`Which composer has the most movies?`)
- **4 hard**: multi-join + subqueries (`Top 5 composers by total songs`)

Each case includes `expected_answer`, `expected_tools`, and `min_steps` for scoring.

---

## Key Design Decisions

- **MemorySaver** (in-memory) — self-contained, no Postgres required
- **asyncio.create_task()** — consistent with finagent/codesentinel pattern
- **LLM-as-judge** for semantic correctness — deterministic scoring can't evaluate natural language answers
- **In-memory report store** — `dict[run_id, EvalReport]` in `app.state`, sufficient for a demo
- **`run_query` read-only guard** — blocks non-SELECT queries at the tool level
