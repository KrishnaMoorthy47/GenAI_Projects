# GenAI Projects

A portfolio of 9 production-grade AI engineering projects — each a standalone FastAPI service covering multi-agent systems, RAG, evaluation frameworks, voice AI, browser automation, and document extraction.

## Projects

| Project | Port | Description | Key Tech |
|---------|------|-------------|----------|
| [`finagent`](./finagent) | 8000 | Multi-agent stock research with human-in-the-loop approval gate | LangGraph, yfinance, SEC EDGAR, Tavily, Postgres |
| [`codesentinel`](./codesentinel) | 8001 | GitHub webhook PR reviewer — parallel security + quality agents | LangGraph, PyGithub, HMAC, SQLite |
| [`chatbot`](./chatbot) | 8002 | RAG chatbot with 11-step pipeline over custom documents | FAISS / pgvector, Redis, PDF ingestion, Groq vision |
| [`agenteval`](./agenteval) | 8003 | ReAct SQL agent + 4-scorer eval framework (task success, tool accuracy, trajectory, hallucination) | LangGraph, SQLite Chinook, LangSmith |
| [`voiceagent`](./voiceagent) | 8004 | Real-time voice AI over WebSocket — speech in, speech out | Whisper STT, Groq LLM, OpenAI TTS, WebSocket |
| [`Job_search_agents`](./Job_search_agents) | 8007 | Browser agents search company career sites, score listings against your CV | browser-use, PyPDF2 |
| [`Stock_analysis_AI_agents`](./Stock_analysis_AI_agents) | 8005 | Dual-engine stock analysis: CrewAI (automated) or LangGraph (human-in-the-loop) | CrewAI, LangGraph, yfinance |
| [`pdf_extraction`](./pdf_extraction) | 8008 | Invoice / PO extraction via LLM Whisperer OCR + structured LLM parsing | LLMWhisperer, Pydantic structured output |
| [`meal-planning-agent`](./meal-planning-agent) | 8006 | 4-agent family meal planner with weekly plan + grocery list, backed by MongoDB | LangGraph, Motor, MongoDB Atlas |

## Shared Stack

- **Framework:** FastAPI + Pydantic v2
- **Agents:** LangGraph, CrewAI, browser-use
- **LLMs:** Groq, OpenAI, Azure OpenAI (switchable via `LLM_PROVIDER` env var)
- **Auth:** `x-api-key` header on all endpoints
- **Rate limiting:** sliding-window middleware (`rate_limit.py`) — 20 req/60s per IP
- **Deployment:** Docker + docker-compose for every project

## Quickstart

All projects use [`uv`](https://github.com/astral-sh/uv). Run from inside each project directory:

```bash
# 1. Set up environment
uv venv && source .venv/bin/activate
uv pip install -r requirements.txt

# 2. Configure secrets
cp .env.example .env   # fill in your API keys

# 3. Run (example ports shown)
PYTHONPATH=src uv run uvicorn finagent.main:app --reload --port 8000       # app/ layout
PYTHONPATH=src uv run uvicorn chatbot.main:app --reload --port 8002        # src/ layout

# Or with Docker
docker-compose -f docker/docker-compose.yml up --build
```

## Environment Setup

Every project reads from a shared `.env` file at the repo root. Minimum required variables:

```bash
API_KEY=your-secret          # x-api-key header value
LLM_PROVIDER=groq            # groq | openai | azure_openai
GROQ_API_KEY=gsk_...         # or OPENAI_API_KEY / AZURE_OPENAI_API_KEY
```

Each project's `ARCHITECTURE.md` lists all variables it needs.
