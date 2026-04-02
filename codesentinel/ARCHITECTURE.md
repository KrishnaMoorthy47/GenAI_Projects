# CodeSentinel — Architecture

LangGraph PR reviewer triggered by GitHub webhooks. A security agent (OWASP regex + LLM) and a quality agent run in parallel, their findings are merged, and a summary agent posts a structured review comment directly to the GitHub PR.

## Flow

```
GitHub Webhook          Manual API
POST /webhook/github    POST /review
      │  (HMAC verify)        │
      └──────────┬────────────┘
                 │  asyncio.create_task()
                 ▼
          parse_diff
         /          \
  security_agent   quality_agent   ← parallel superstep
  (OWASP + LLM)    (LLM)
         \          /
          merge_findings
                 │
          summary_agent
                 │
        POST GitHub PR review
```

## API

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| GET | `/health` | No | Health check |
| POST | `/webhook/github` | HMAC | GitHub webhook receiver |
| POST | `/review` | Yes | Manual review trigger |
| GET | `/review/{id}` | Yes | Poll review status and results |

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `API_KEY` | required | `x-api-key` header value |
| `LLM_PROVIDER` | `groq` | `groq` \| `openai` \| `azure_openai` |
| `GROQ_API_KEY` | — | Groq API key |
| `OPENAI_API_KEY` | — | OpenAI API key |
| `GITHUB_TOKEN` | — | PAT for posting PR reviews |
| `GITHUB_WEBHOOK_SECRET` | — | HMAC secret for webhook verification |
| `DATABASE_URL` | — | SQLite path for review persistence |
