# FinAgent — Runbook

## Prerequisites

| Tool | Version | Install |
|------|---------|---------|
| Python | 3.12+ | [python.org](https://python.org) |
| uv | latest | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| Docker Desktop | latest | [docker.com](https://docker.com) |
| OpenAI API key | — | [platform.openai.com](https://platform.openai.com) |
| Tavily API key | — | [tavily.com](https://app.tavily.com) (free tier available) |

---

## Step 1 — Fill in `.env`

Open `finagent/.env` and add your keys:

```
API_KEY=any-secret-string-you-choose    ← used in x-api-key header
OPENAI_API_KEY=sk-...                   ← from platform.openai.com
TAVILY_API_KEY=tvly-...                 ← from app.tavily.com
POSTGRES_PASSWORD=finagent              ← leave as-is for local dev
```

---

## Option A — Docker Compose (recommended, no Python install needed)

```bash
cd finagent

# Build and start postgres + app
docker compose -f docker/docker-compose.yml up --build -d

# Tail logs
docker compose -f docker/docker-compose.yml logs -f app

# Verify the app is ready
curl http://localhost:8000/health
# Expected: {"status": "ok", "service": "finagent"}
```

---

## Option B — Local (Python + uv)

```bash
cd finagent

# Create virtual env and install deps
uv venv && source .venv/bin/activate
uv pip install -r requirements.txt

# Start Postgres only
docker compose -f docker/docker-compose.yml up -d postgres

# Wait ~5s for Postgres to be ready, then start the app
uvicorn finagent.main:app --reload --port 8000
```

---

## Step 2 — Smoke Test (end-to-end)

Run these in order. Each step depends on the previous one.

### 2a. Health check
```bash
curl http://localhost:8000/health
```
Expected:
```json
{"status": "ok", "service": "finagent"}
```

### 2b. Start research
```bash
curl -X POST http://localhost:8000/research \
  -H "x-api-key: YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"ticker": "AAPL"}'
```
Expected:
```json
{
  "thread_id": "abc-123-...",
  "status": "started",
  "ticker": "AAPL",
  "message": "Research started for AAPL. Stream events at GET /research/abc-123-.../stream"
}
```
**Copy the `thread_id`** — you'll use it in the next steps.

### 2c. Stream live events (open in a separate terminal)
```bash
curl -N http://localhost:8000/research/YOUR_THREAD_ID/stream \
  -H "x-api-key: YOUR_API_KEY"
```
You'll see events like:
```
data: {"type": "node_complete", "node": "web_research"}
data: {"type": "node_complete", "node": "financial_data"}
data: {"type": "node_complete", "node": "sentiment"}
data: {"type": "node_complete", "node": "report_writer"}
data: {"type": "awaiting_approval", "message": "..."}
```

### 2d. Check status
```bash
curl http://localhost:8000/research/YOUR_THREAD_ID/status \
  -H "x-api-key: YOUR_API_KEY"
```
Expected (after streaming completes):
```json
{"thread_id": "...", "status": "awaiting_approval", "ticker": "AAPL"}
```

### 2e. Approve the report
```bash
curl -X POST http://localhost:8000/research/YOUR_THREAD_ID/approve \
  -H "x-api-key: YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"approved": true}'
```
Expected:
```json
{"thread_id": "...", "status": "completed", "message": "Research approved and finalized."}
```

### 2f. Get the final report
```bash
curl http://localhost:8000/research/YOUR_THREAD_ID/report \
  -H "x-api-key: YOUR_API_KEY"
```
Expected: full `InvestmentReport` JSON with recommendation, thesis, financials, risks, etc.

---

## Step 3 — Run Tests

```bash
cd finagent
source .venv/bin/activate  # or activate your venv

uv run pytest tests/ -v
```

Expected output:
```
tests/test_tools.py::TestCalculator::test_basic_arithmetic PASSED
tests/test_tools.py::TestCalculator::test_percentage_calculation PASSED
tests/test_tools.py::TestCalculator::test_sqrt PASSED
tests/test_tools.py::TestCalculator::test_division_by_zero PASSED
tests/test_tools.py::TestCalculator::test_unsafe_expression_rejected PASSED
tests/test_tools.py::TestFinanceTools::test_get_stock_info_returns_json PASSED
tests/test_tools.py::TestFinanceTools::test_get_stock_info_handles_error PASSED
tests/test_tools.py::TestFinanceTools::test_get_earnings_history_empty PASSED
tests/test_tools.py::TestWebSearchTool::test_web_search_formats_results PASSED
tests/test_tools.py::TestWebSearchTool::test_web_search_handles_error PASSED
tests/test_graph.py::TestGraphCompilation::test_graph_compiles_with_mock_checkpointer PASSED
tests/test_graph.py::TestGraphCompilation::test_graph_has_expected_nodes PASSED
tests/test_graph.py::TestSupervisorRouting::test_routes_to_web_research_first PASSED
tests/test_graph.py::TestSupervisorRouting::test_routes_to_financial_data_after_web PASSED
tests/test_graph.py::TestSupervisorRouting::test_routes_to_sentiment_after_both_research PASSED
tests/test_graph.py::TestSupervisorRouting::test_routes_to_report_writer_when_all_done PASSED
tests/test_graph.py::TestStreamingService::test_create_and_get_queue PASSED
tests/test_graph.py::TestStreamingService::test_sse_generator_returns_error_for_unknown_thread PASSED
```

---

## Step 4 — View API Docs (Swagger UI)

Open in browser: **http://localhost:8000/docs**

All 5 endpoints are interactive — you can test them directly from the browser.

---

## Teardown

```bash
# Stop and remove containers + Postgres volume
docker compose -f docker/docker-compose.yml down -v
```

---

## Known Limitations

### 1. No authentication beyond API key
The single static `x-api-key` is sufficient for personal/demo use but **not production-ready**. A production deployment would need OAuth2, JWT tokens, rate limiting per user, and IP allowlisting.

### 2. In-memory stream registry lost on restart
`asyncio.Queue` objects in `streaming.py` live in the process memory. If the app restarts mid-research, the SSE stream is gone. The graph state is safe in Postgres (checkpointed), but you'd need to poll `/status` instead of streaming after a restart.

### 3. No concurrent research sessions per ticker
Two simultaneous requests for the same ticker create independent `thread_id`s with independent Postgres checkpoints. This is correct behavior, but doubles API costs. No deduplication is implemented.

### 4. SEC EDGAR rate limits
The EDGAR API (`data.sec.gov`) has a soft rate limit of ~10 requests/second. Heavy use may get throttled (HTTP 429). The current implementation has no retry/backoff for EDGAR calls.

### 5. yfinance data freshness
`yfinance` scrapes Yahoo Finance and is **not an official API**. Data can be delayed (15–20 min for price), occasionally unavailable, or return `None` for some tickers. All `yf.Ticker()` calls are wrapped in `try/except`.

### 6. Tavily API costs
Tavily's free tier gives ~1000 searches/month. Each research session makes 4–8 searches. At scale, this becomes a cost concern.

### 7. LLM hallucination risk in financial context
The investment reports are **AI-generated and not financial advice**. The LLM may confidently state inaccurate numbers. Always verify against primary sources before any real decisions.

### 8. Postgres required for persistence
There is no SQLite fallback. Without Postgres running, the app fails on startup. For a demo without Docker, you'd need to change the checkpointer to `MemorySaver` (loses persistence across restarts).

### 9. Streaming not resumable
If a client disconnects during SSE streaming (e.g., network drop), the graph continues running in the background but the SSE events emitted during disconnection are lost. The client must reconnect and wait for the next event (or poll `/status`).

### 10. Single-worker only
`uvicorn --workers 1` is required because the `asyncio.Queue` stream registry is in-process memory. Multiple workers would not share the queue, causing stream-not-found errors. For multi-worker deployment, the queue must be replaced with Redis pub/sub.
