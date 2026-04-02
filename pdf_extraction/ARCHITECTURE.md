# PDF Extraction — Architecture

AI-powered document extraction service for invoices and purchase orders. PDFs are processed through LLM Whisperer OCR (handles scanned documents, handwriting, and complex layouts), then an LLM performs structured field extraction against a Pydantic schema, returning a typed JSON object with header info and all product line items.

## Flow

```
POST /api/v1/extract  (PDF upload)
      │
      ▼  asyncio.create_task()
┌─────────────────────────────────┐
│  Extraction Pipeline            │
│                                 │
│  PDF bytes → LLM Whisperer OCR  │
│       → raw text                │
│       → LLM with_structured_output()
│       → ExtractionResult JSON   │
└─────────────────────────────────┘
      │
  in-memory job store
      │
GET /api/v1/extract/{id}
```

## API

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| GET | `/health` | No | Service health check |
| POST | `/api/v1/extract` | Yes | Upload PDF, start extraction job |
| GET | `/api/v1/extract/{id}` | Yes | Poll status and retrieve results |

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `API_KEY` | required | `x-api-key` header value |
| `LLM_PROVIDER` | `groq` | `groq` \| `openai` \| `azure_openai` |
| `GROQ_API_KEY` | — | Groq API key |
| `OPENAI_API_KEY` | — | OpenAI API key |
| `LLMWHISPERER_API_KEY` | — | LLM Whisperer API key |
| `LLMWHISPERER_BASE_URL` | — | LLM Whisperer endpoint |
| `PORT` | `8008` | Server port |
