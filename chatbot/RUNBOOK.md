# Personal RAG Chatbot — Runbook

## Prerequisites

| Tool | Version | Install |
|------|---------|---------|
| Python | 3.11+ | [python.org](https://python.org) |
| uv | latest | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| Docker Desktop | latest | [docker.com](https://docker.com) — needed for Redis |
| OpenAI API key | — | [platform.openai.com](https://platform.openai.com) — for GPT-4o + embeddings |

**Free alternative — Ollama (no API key needed for LLM):**
```bash
# Install from https://ollama.ai, then:
ollama pull llama3.2
# Note: You still need OPENAI_API_KEY for embeddings (text-embedding-3-small)
```

---

## Step 1 — Fill in `.env`

Open `chatbot/.env` and add your keys:

```
API_KEY=any-secret-string-you-choose    ← used in x-api-key header
OPENAI_API_KEY=sk-...                   ← from platform.openai.com
```

For Ollama (free LLM, still needs OpenAI for embeddings):
```
LLM_PROVIDER=ollama
LLM_MODEL=llama3.2
OPENAI_API_KEY=sk-...                   ← still needed for embeddings
```

---

## Step 2 — Add Your Documents

Put any PDF, TXT, MD, or HTML files in `chatbot/data/input/`:

```bash
# Examples:
cp ~/Downloads/user-manual.pdf chatbot/data/input/
cp ~/notes/faq.txt chatbot/data/input/
cp ~/docs/policy.md chatbot/data/input/
```

You need at least one document before ingesting.

---

## Option A — Docker Compose (recommended)

```bash
cd chatbot

# Start Redis + app
docker compose up --build -d

# Ingest your documents (run this after adding files to data/input/)
docker compose exec app python scripts/ingest.py

# Tail logs
docker compose logs -f app

# Verify
curl http://localhost:8000/api/v1/health
```
Expected:
```json
{"status": "healthy", "version": "1.0.0", "index_loaded": true, "redis_connected": true}
```

---

## Option B — Local (Python + uv)

```bash
cd chatbot

# Create and activate virtual env
uv venv && source .venv/bin/activate
uv pip install -r requirements.txt

# Start Redis (for chat history — app works without it, falls back to in-memory)
docker run -d -p 6379:6379 --name chatbot-redis redis:7.2-alpine

# Run ingestion (before starting the app)
PYTHONPATH=src python scripts/ingest.py

# Start the app
PYTHONPATH=src uvicorn chatbot.main:app --reload --port 8000
```

---

## Step 3 — Ingestion Script (standalone)

```bash
# Default: reads data/input/, writes to data/indexed/default/
PYTHONPATH=src python scripts/ingest.py

# Custom paths
PYTHONPATH=src python scripts/ingest.py \
  --input /path/to/my/docs \
  --output /path/to/index \
  --collection my-docs
```

Expected output:
```
============================================================
  Personal RAG Chatbot — Document Ingestion
============================================================

✓ Ingestion complete.
  Documents processed : 3
  Total chunks created: 412
  Vectors indexed     : 412
  Index saved to      : data/indexed/default
  Time elapsed        : 18.4s
```

**Re-run ingestion whenever you add or update documents.** The old index is replaced completely.

---

## Step 4 — Smoke Test (end-to-end)

### 4a. Health check
```bash
curl http://localhost:8000/api/v1/health
```
Expected:
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "index_loaded": true,
  "redis_connected": true
}
```
If `index_loaded: false` → run the ingestion script first.
If `redis_connected: false` → Redis is down, using in-memory fallback (fine for testing).

### 4b. Ask a question about your documents
```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "x-api-key: YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the return policy?",
    "session_id": "test-session-001"
  }'
```
Expected:
```json
{
  "answer": "According to the policy document, returns are accepted within 30 days...",
  "session_id": "test-session-001",
  "sources": [
    {"filename": "policy.pdf", "page_num": 3}
  ],
  "has_context": true
}
```

If `has_context: false` → the question topic wasn't found in your documents.

### 4c. Multi-turn conversation (same session_id)
```bash
# Turn 2 — chatbot remembers turn 1
curl -X POST http://localhost:8000/api/v1/query \
  -H "x-api-key: YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What about digital purchases?",
    "session_id": "test-session-001"
  }'
```
The chatbot will use conversation history from turn 1 to understand the follow-up.

### 4d. Test with a question outside your documents
```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "x-api-key: YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Who won the 2024 World Cup?",
    "session_id": "test-session-002"
  }'
```
Expected: `has_context: false` with a "I don't have relevant information" answer — no hallucination.

---

## Step 5 — Run Tests

```bash
cd chatbot
source .venv/bin/activate

PYTHONPATH=src uv run pytest tests/ -v
```

Expected output:
```
tests/test_rag_pipeline.py::TestQuerySanitization::test_strips_whitespace PASSED
tests/test_rag_pipeline.py::TestQuerySanitization::test_collapses_spaces PASSED
tests/test_rag_pipeline.py::TestQuerySanitization::test_removes_control_chars PASSED
tests/test_rag_pipeline.py::TestQuerySanitization::test_raises_on_empty PASSED
tests/test_rag_pipeline.py::TestContextTokenBudget::test_fits_all_chunks_within_budget PASSED
tests/test_rag_pipeline.py::TestContextTokenBudget::test_excludes_chunks_over_budget PASSED
tests/test_rag_pipeline.py::TestDynamicThreshold::test_filters_low_scoring_chunks PASSED
tests/test_rag_pipeline.py::TestDynamicThreshold::test_all_pass_when_scores_are_close PASSED
tests/test_rag_pipeline.py::TestSchemaValidation::test_valid_session_id PASSED
tests/test_rag_pipeline.py::TestSchemaValidation::test_invalid_session_id_too_short PASSED
tests/test_rag_pipeline.py::TestSchemaValidation::test_empty_question_rejected PASSED
tests/test_rag_pipeline.py::TestCacheAdapter::test_in_memory_get_returns_empty_for_new_session PASSED
tests/test_rag_pipeline.py::TestCacheAdapter::test_in_memory_save_and_retrieve PASSED
tests/test_rag_pipeline.py::TestCacheAdapter::test_history_is_cleared PASSED
tests/test_ingestion.py::TestDiscoverDocuments::test_finds_supported_extensions PASSED
tests/test_ingestion.py::TestDiscoverDocuments::test_returns_empty_for_empty_dir PASSED
tests/test_ingestion.py::TestLoadAndChunk::test_chunks_text_file PASSED
tests/test_ingestion.py::TestLoadAndChunk::test_skips_unreadable_file PASSED
tests/test_ingestion.py::TestVectorAdapter::test_save_and_load_index PASSED
tests/test_ingestion.py::TestVectorAdapter::test_load_returns_false_for_missing_index PASSED
tests/test_ingestion.py::TestVectorAdapter::test_is_index_loaded_false_before_load PASSED
```

---

## Step 6 — View API Docs

Open: **http://localhost:8000/docs**

Two interactive endpoints are available to test from the browser.

---

## Teardown

```bash
# Docker Compose
docker compose down -v

# Local Redis
docker stop chatbot-redis && docker rm chatbot-redis
```

---

## Common Issues

| Problem | Cause | Fix |
|---------|-------|-----|
| `index_loaded: false` | Ingestion not run yet | Run `python scripts/ingest.py` |
| `has_context: false` | Question not in docs | Ask about topics actually in your documents |
| `401 Unauthorized` | Wrong API key | Check `API_KEY` in `.env` matches header |
| `ImportError: faiss` | FAISS not installed | `uv pip install faiss-cpu` |
| `redis_connected: false` | Redis not running | App still works (in-memory fallback), or `docker run -d -p 6379:6379 redis:7.2-alpine` |
| Slow first query | Cold embedding call | Expected — subsequent queries are faster (FAISS cached) |

---

## Known Limitations

### 1. FAISS index is not incremental
Every time you run `ingest.py`, the entire index is rebuilt from scratch. Adding one new document requires re-embedding all existing documents. For large collections (>10k chunks), this becomes slow. A production system would use an incremental vector database (Qdrant, Pinecone, Weaviate).

### 2. No streaming responses
The API is synchronous — the full LLM response is generated before returning. For long answers, the client waits 3–10 seconds. Adding `StreamingResponse` with SSE would improve perceived latency.

### 3. In-memory FAISS — no persistence across Docker restarts without volume mount
The FAISS index is loaded from disk (`data/indexed/`) into memory at startup. The `data/` directory must be volume-mounted in Docker (`./data:/app/data`) to persist the index across container restarts. Without the mount, you'd need to re-ingest on every restart.

### 4. Single collection at a time per process
The default collection (`DEFAULT_COLLECTION=default`) is pre-loaded at startup. Querying a different collection (`collection_name` in the request) works, but that collection's index is loaded on-demand and cached. There's no eviction — all collections stay in memory indefinitely.

### 5. Embedding model locked to OpenAI
Embeddings use `text-embedding-3-small` (OpenAI). If you switch to Ollama for LLM, you still need an OpenAI API key for embeddings. There's no built-in support for local embedding models (e.g., `nomic-embed-text` via Ollama). Adding Ollama embeddings would require modifying `llm_adapter.py`.

### 6. No document update tracking
There's no checksum or timestamp tracking of ingested files. If you modify a document and re-ingest, the old chunks are replaced, but there's no way to ingest only changed files — the full pipeline always runs.

### 7. PDF scanned images not supported
`PyPDFLoader` extracts text from text-layer PDFs. Scanned PDFs (image-only, no text layer) return no content. You'd need OCR (e.g., `pytesseract`, `unstructured`) to handle those.

### 8. Chat history not portable across restarts (without Redis)
When using the in-memory fallback (no Redis), all chat history is lost on app restart. With Redis, history persists for `CACHE_TTL` seconds (default: 1 hour) after the last message.

### 9. No user isolation
All sessions share the same FAISS index and the same Redis namespace. There's no per-user access control — any caller with the API key can query any collection.

### 10. Content safety check skipped for Ollama
The OpenAI Moderation API is only called when `LLM_PROVIDER=openai`. When using Ollama, `moderate()` returns `True` (safe) immediately without any check. If you need content safety with Ollama, integrate a local moderation model separately.
