# Personal RAG Chatbot

A production-grade **Retrieval-Augmented Generation (RAG)** chatbot that answers questions over your own documents. Drop in PDFs, text files, or HTML pages — the chatbot retrieves the most relevant passages and answers using GPT-4o (or a local Ollama model).

## How It Works

```
Your documents (PDF/TXT/HTML)
        │
        ▼
  scripts/ingest.py
  ┌─────────────────────────────────┐
  │ 1. Discover documents           │
  │ 2. Chunk with 70% overlap       │
  │ 3. Embed (text-embedding-3-small)│
  │ 4. Save FAISS index to disk     │
  └─────────────────────────────────┘
        │
        ▼
  data/indexed/default/
  (index.faiss + index.pkl)

        │ (loaded at startup)
        ▼

POST /api/v1/query
  ┌─────────────────────────────────────────────────────┐
  │  1. Validate API key (x-api-key header)             │
  │  2. Sanitize query                                  │
  │  3. Content safety check (OpenAI Moderation API)   │
  │  4. Embed query → FAISS similarity search (top-5)  │
  │  5. Dynamic threshold filter (keep score ≥ 80% max)│
  │  6. Build context block (token budget: 4000 tokens)│
  │  7. Load chat history from Redis                   │
  │  8. (Optional) Relevance pre-check                 │
  │  9. LLM call (system + history + context + query)  │
  │ 10. Save updated history to Redis                  │
  │ 11. Return answer + source citations               │
  └─────────────────────────────────────────────────────┘
```

## Quick Start

### 1. Docker Compose (recommended)

```bash
cp .env.example .env
# Edit .env — set OPENAI_API_KEY and API_KEY

# Start Redis + app
docker compose up --build -d

# Drop your documents into data/input/
cp ~/your-docs/*.pdf data/input/

# Run ingestion (once, or whenever documents change)
docker compose exec app python scripts/ingest.py

# Query!
curl -X POST http://localhost:8000/api/v1/query \
  -H "x-api-key: your_secret_api_key_here" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the return policy?", "session_id": "my-session-001"}'
```

### 2. Local development

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env  # Fill in OPENAI_API_KEY and API_KEY

# Start Redis (or set REDIS_HOST to skip — falls back to in-memory)
docker run -d -p 6379:6379 redis:7.2-alpine

# Add documents
cp ~/your-docs/*.pdf data/input/

# Ingest
PYTHONPATH=src python scripts/ingest.py

# Run the app
PYTHONPATH=src uvicorn chatbot.main:app --reload --port 8000
```

### 3. Use with Ollama (free, local, no API costs)

```bash
# Install Ollama: https://ollama.ai
ollama pull llama3.2

# Set in .env:
# LLM_PROVIDER=ollama
# LLM_MODEL=llama3.2
# OLLAMA_BASE_URL=http://localhost:11434
# Note: Embedding still requires OPENAI_API_KEY (or swap to a local embedding model)
```

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `API_KEY` | Yes | — | `x-api-key` header for auth. Generate: `python -c "import secrets; print(secrets.token_hex(32))"` |
| `LLM_PROVIDER` | Yes | `openai` | `openai` or `ollama` |
| `OPENAI_API_KEY` | If OpenAI | — | OpenAI API key |
| `LLM_MODEL` | No | `gpt-4o` | Model name (e.g. `gpt-4o`, `llama3.2`) |
| `LLM_MAX_TOKENS` | No | `1500` | Max tokens in LLM response |
| `LLM_TEMPERATURE` | No | `0.2` | 0 = deterministic, 1 = creative |
| `EMBEDDING_MODEL` | No | `text-embedding-3-small` | OpenAI embedding model |
| `OLLAMA_BASE_URL` | If Ollama | `http://localhost:11434` | Ollama server URL |
| `DATA_INPUT_DIR` | No | `data/input` | Where to look for documents |
| `DATA_INDEXED_DIR` | No | `data/indexed` | Where to save the FAISS index |
| `DEFAULT_COLLECTION` | No | `default` | Index collection name |
| `VECTOR_SEARCH_K` | No | `5` | Number of chunks to retrieve |
| `MAX_CONTEXT_TOKENS` | No | `4000` | Token budget for retrieved context |
| `THRESHOLD_RATIO` | No | `0.8` | Keep chunks scoring ≥ this × best score |
| `MAX_QUERY_LENGTH` | No | `1000` | Max characters per question |
| `ENABLE_RELEVANCE_PRECHECK` | No | `false` | Extra LLM call to verify context relevance |
| `MAX_HISTORY_TURNS` | No | `5` | Conversation turns to remember |
| `CACHE_TTL` | No | `3600` | Session expiry in seconds (1 hour) |
| `REDIS_HOST` | No | `localhost` | Redis hostname (`redis` inside Docker) |
| `REDIS_PORT` | No | `6379` | Redis port |
| `REDIS_PASSWORD` | No | — | Redis password (leave blank for local dev) |
| `ENABLE_CONTENT_SAFETY` | No | `true` | Run OpenAI moderation on inputs |
| `CHUNK_SIZE` | No | `575` | Tokens per document chunk |
| `CHUNK_OVERLAP` | No | `400` | Token overlap between chunks (~70%) |

## API Reference

### POST /api/v1/query

Answer a question over your documents.

**Auth:** `x-api-key` header required.

**Request:**
```json
{
  "question": "What is the return policy?",
  "session_id": "my-session-001",
  "collection_name": "default",
  "language": "en"
}
```

**Response:**
```json
{
  "answer": "According to the policy document, returns are accepted within 30 days of purchase...",
  "session_id": "my-session-001",
  "sources": [
    {"filename": "return_policy.pdf", "page_num": 3},
    {"filename": "faq.txt", "page_num": 1}
  ],
  "has_context": true
}
```

**Status Codes:**
- `200` — Success (check `has_context` to know if documents were used)
- `400` — Invalid request or query too long
- `401` — Invalid API key
- `422` — Pydantic validation error
- `451` — Content policy violation
- `500` — Internal error

### GET /api/v1/health

No auth required.

```json
{
  "status": "healthy",
  "version": "1.0.0",
  "index_loaded": true,
  "redis_connected": true
}
```

## Supported Document Formats

| Extension | Loader |
|-----------|--------|
| `.pdf` | PyPDFLoader (page-aware) |
| `.txt` | TextLoader |
| `.md` | TextLoader |
| `.html` / `.htm` | BSHTMLLoader (strips tags) |

## Multi-turn Conversation

The `session_id` ties turns together. Use the same ID across requests:

```bash
# Turn 1
curl -X POST http://localhost:8000/api/v1/query \
  -H "x-api-key: $API_KEY" -H "Content-Type: application/json" \
  -d '{"question": "What is the return policy?", "session_id": "conv-001"}'

# Turn 2 (chatbot remembers context from turn 1)
curl -X POST http://localhost:8000/api/v1/query \
  -H "x-api-key: $API_KEY" -H "Content-Type: application/json" \
  -d '{"question": "What about digital purchases?", "session_id": "conv-001"}'
```

## Ingestion Script

```bash
# Default — uses settings from .env
python scripts/ingest.py

# Custom paths
python scripts/ingest.py --input /path/to/docs --output /path/to/index --collection my-docs
```

Output:
```
============================================================
  Personal RAG Chatbot — Document Ingestion
============================================================

✓ Ingestion complete.
  Documents processed : 5
  Total chunks created: 847
  Vectors indexed     : 847
  Index saved to      : data/indexed/default
  Time elapsed        : 43.2s
```

## Running Tests

```bash
PYTHONPATH=src pytest tests/ -v --cov=src/chatbot
```

## Architecture Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Vector store | FAISS `IndexFlatIP` | Free, no server, exact search, cosine similarity with L2 normalization |
| Chunking | `TokenTextSplitter`, 575 tokens, 70% overlap | Preserves context at boundaries; technical docs benefit from high overlap |
| Threshold | Dynamic (80% of best score) | Prevents LLM hallucination when query topic isn't in the docs |
| History | Redis + in-memory fallback | Persistent across requests; gracefully degrades without Redis |
| LLM temperature | 0.2 | Factual, reproducible answers with slight flexibility |
| Content safety | OpenAI Moderation API | Free endpoint, same API key, no extra setup |

## Tech Stack

- **[FastAPI](https://fastapi.tiangolo.com)** — Async REST API
- **[FAISS](https://github.com/facebookresearch/faiss)** — Vector similarity search
- **[LangChain](https://python.langchain.com)** — Document loaders and text splitters
- **[OpenAI](https://platform.openai.com)** — GPT-4o LLM + embeddings + moderation
- **[Ollama](https://ollama.ai)** — Local LLM alternative (no API cost)
- **[Redis](https://redis.io)** — Chat history persistence
- **[tiktoken](https://github.com/openai/tiktoken)** — Token counting for context budgeting
