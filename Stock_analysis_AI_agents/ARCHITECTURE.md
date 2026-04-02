# Stock Intelligence — Architecture

Dual-engine stock analysis service. Choose **CrewAI** for a fully-automated 4-agent crew that produces a BUY/HOLD/SELL recommendation, or **LangGraph** for a human-in-the-loop workflow with an approval gate before the final report is published. Both engines use yfinance for real market data.

## Flow

```
POST /api/v1/analyze  (symbol + engine)
      │
      ├── engine=crewai ──► CrewAI crew (researcher → analyst → risk_mgr → strategist)
      │                          │
      │                     yfinance tools
      │                          │
      │                     BUY/HOLD/SELL report
      │
      └── engine=langgraph ──► supervisor → [web_research ‖ financial_data ‖ sentiment]
                                    │
                               report_writer
                                    │
                               human_review ◄── INTERRUPT
                                    │
                          POST /approve → resume → final report
```

## API

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| GET | `/health` | No | Service health check |
| POST | `/api/v1/analyze` | Yes | Start analysis (crewai or langgraph) |
| GET | `/api/v1/analyze/{id}` | Yes | Poll status and result |
| POST | `/api/v1/analyze/{id}/approve` | Yes | Approve or reject LangGraph report |
| GET | `/api/v1/stock/{symbol}/quote` | Yes | Real-time yfinance quote |
| GET | `/api/v1/stocks/popular` | Yes | Curated list of popular stocks |

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `API_KEY` | required | `x-api-key` header value |
| `LLM_PROVIDER` | `groq` | `groq` \| `openai` \| `azure_openai` |
| `GROQ_API_KEY` | — | Groq API key |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | Groq model |
| `OPENAI_API_KEY` | — | OpenAI API key |
| `TAVILY_API_KEY` | — | Web search (LangGraph engine) |
| `DATABASE_URL` | — | Postgres DSN (LangGraph checkpointer) |
| `PORT` | `8005` | Server port |
