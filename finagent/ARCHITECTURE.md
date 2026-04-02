# FinAgent — Architecture

LangGraph multi-agent system that researches a stock ticker using three specialist agents (web search, financial data, sentiment), produces a structured investment brief, and pauses for human approval before finalising. Streams progress in real time via SSE.

## Flow

```
POST /research
      │
      ▼
  supervisor ──► web_research  (Tavily)
      ▲               │
      └───────── supervisor ──► financial_data  (yfinance + SEC EDGAR)
                      ▲               │
                      └───────── supervisor ──► sentiment  (LLM)
                                      │
                                      ▼
                               report_writer  (structured output)
                                      │
                               human_review ◄── INTERRUPT
                                      │
                             POST /approve → resume
```

## API

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| GET | `/health` | No | Health check |
| POST | `/research` | Yes | Start stock research |
| GET | `/research/{id}/stream` | Yes | SSE stream of progress events |
| GET | `/research/{id}/status` | Yes | Current status |
| POST | `/research/{id}/approve` | Yes | Approve or reject the report |
| GET | `/research/{id}/report` | Yes | Retrieve completed report |

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `API_KEY` | required | `x-api-key` header value |
| `LLM_PROVIDER` | `groq` | `groq` \| `openai` \| `azure_openai` |
| `GROQ_API_KEY` | — | Groq API key |
| `OPENAI_API_KEY` | — | OpenAI API key |
| `TAVILY_API_KEY` | — | Web search |
| `DATABASE_URL` | — | Postgres DSN for LangGraph checkpointer |
