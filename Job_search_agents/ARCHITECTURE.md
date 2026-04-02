# Job Search Agent — Architecture

AI-powered job search service that automates browsing company career sites. Upload a CV (PDF) and specify target companies — the system spawns parallel browser-use agents that navigate each company's careers page, extract job listings, score each role against the CV using an LLM, and return results ranked by fit score.

## Flow

```
POST /api/v1/search  (CV PDF + companies)
      │
      ▼
  cv_parser (PyPDF2)
      │
      ▼  asyncio.create_task()
  browser_agent service
      │
  ┌───┴──────────────────┐
  ▼                      ▼
browser-use agent    browser-use agent   ← one per company, parallel
(Company A careers)  (Company B careers)
      │
      ▼
  LLM scorer  (0–100 fit score vs CV)
      │
      ▼
  in-memory search store
      │
GET /api/v1/search/{id}/results
```

## API

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| GET | `/health` | No | Service health check |
| GET | `/api/v1/companies` | Yes | List supported company career sites |
| POST | `/api/v1/search` | Yes | Upload CV + start parallel job search |
| GET | `/api/v1/search/{id}/status` | Yes | Poll search progress |
| GET | `/api/v1/search/{id}/results` | Yes | Retrieve ranked job listings |

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `API_KEY` | required | `x-api-key` header value |
| `LLM_PROVIDER` | `groq` | `groq` \| `openai` \| `azure_openai` |
| `GROQ_API_KEY` | — | Groq API key |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | Groq model |
| `OPENAI_API_KEY` | — | OpenAI API key |
| `CHROME_PATH` | Windows default | Path to Chrome executable |
| `HEADLESS` | `true` | Run browser headless |
| `PORT` | `8007` | Server port |
