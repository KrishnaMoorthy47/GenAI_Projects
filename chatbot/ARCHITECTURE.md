# RAG Chatbot — Architecture

RAG chatbot that answers questions over your own documents (PDF, TXT, Markdown, HTML). Documents are ingested once into a FAISS vector index; queries go through an 11-step pipeline — sanitise, moderate, embed, retrieve, filter, budget, load history, check relevance, call LLM, save history, return. Built with hexagonal architecture so the LLM, vector store, and cache are all swappable via env vars.

## Flow

```
POST /api/v1/upload          POST /api/v1/query
      │                              │
      ▼                              ▼
ingestion_service            query_service (11-step pipeline)
      │                              │
      ├── extract text               ├── sanitize → moderate
      ├── chunk (TokenSplitter)      ├── embed → FAISS search
      ├── embed (OpenAI)             ├── threshold filter
      └── FAISS index                ├── token budget
            │                       ├── load history (Redis)
      data/indexed/                  ├── LLM call (OpenAI / Ollama)
                                     └── save history → return
                                              │
                               llm_adapter / vector_adapter / cache_adapter
```

## API

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| GET | `/health` | No | Health check |
| POST | `/api/v1/query` | Yes | Ask a question |
| POST | `/api/v1/upload` | Yes | Ingest a document |

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `API_KEY` | required | `x-api-key` header value |
| `LLM_PROVIDER` | `openai` | `openai` \| `ollama` |
| `OPENAI_API_KEY` | — | OpenAI API key |
| `VECTOR_BACKEND` | `faiss` | `faiss` \| `supabase` |
| `SUPABASE_URL` | — | Supabase project URL (if using pgvector) |
| `SUPABASE_KEY` | — | Supabase anon key |
| `REDIS_URL` | — | Redis for chat history (falls back to in-memory) |
