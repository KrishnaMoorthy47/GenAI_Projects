# CodeSentinel — Runbook

## Prerequisites

| Tool | Version | Install |
|------|---------|---------|
| Python | 3.12+ | [python.org](https://python.org) |
| uv | latest | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| Docker Desktop | latest | [docker.com](https://docker.com) |
| OpenAI API key | — | [platform.openai.com](https://platform.openai.com) |
| GitHub PAT | — | GitHub → Settings → Developer Settings → Personal Access Tokens |
| ngrok (optional) | latest | [ngrok.com](https://ngrok.com) — only for live webhook testing |

### GitHub PAT Required Scopes
When creating your token at `github.com/settings/tokens`:
- `repo` → pull requests read/write (to post review comments)

---

## Step 1 — Fill in `.env`

Open `codesentinel/.env` and add your keys:

```
API_KEY=any-secret-string-you-choose
OPENAI_API_KEY=sk-...
GITHUB_TOKEN=ghp_...                   ← your GitHub PAT
GITHUB_WEBHOOK_SECRET=                 ← leave blank for local dev (skips HMAC check)
```

---

## Option A — Docker Compose (recommended)

```bash
cd codesentinel

docker compose -f docker/docker-compose.yml up --build -d

# Tail logs
docker compose -f docker/docker-compose.yml logs -f app

# Verify
curl http://localhost:8001/health
# Expected: {"status": "ok", "service": "codesentinel"}
```

---

## Option B — Local (Python + uv)

```bash
cd codesentinel

uv venv && source .venv/bin/activate
uv pip install -r requirements.txt

uvicorn codesentinel.main:app --reload --port 8001
```

---

## Step 2 — Smoke Test: Manual API Trigger

This is the easiest way to test without setting up a GitHub webhook.

### 2a. Health check
```bash
curl http://localhost:8001/health
```
Expected:
```json
{"status": "ok", "service": "codesentinel"}
```

### 2b. Trigger a PR review
You need a real GitHub repo and PR number. Use any public repo or your own.

```bash
curl -X POST http://localhost:8001/review \
  -H "x-api-key: YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "repo": "octocat/Hello-World",
    "pr_number": 1,
    "github_token": "ghp_..."
  }'
```
Expected:
```json
{
  "review_id": "abc-123-...",
  "repo": "octocat/Hello-World",
  "pr_number": 1,
  "status": "pending",
  ...
}
```

### 2c. Poll for results
The review runs in the background (takes 10–30s depending on diff size).

```bash
# Poll every few seconds until status = "completed"
curl http://localhost:8001/review/YOUR_REVIEW_ID \
  -H "x-api-key: YOUR_API_KEY"
```
Expected (when done):
```json
{
  "review_id": "...",
  "status": "completed",
  "security_findings": [...],
  "quality_findings": [...],
  "final_review": "## CodeSentinel Review\n\n...",
  "pr_comment_posted": true,
  "critical_count": 0,
  "high_count": 2,
  ...
}
```

The `final_review` field contains the Markdown review that was posted to the GitHub PR.

---

## Step 3 — Test Webhook (optional, needs ngrok)

### 3a. Start ngrok tunnel
```bash
ngrok http 8001
# Copy the Forwarding URL, e.g.: https://abc123.ngrok-free.app
```

### 3b. Register webhook in GitHub
1. Go to your repo → **Settings → Webhooks → Add webhook**
2. Payload URL: `https://abc123.ngrok-free.app/webhook/github`
3. Content type: `application/json`
4. Secret: set `GITHUB_WEBHOOK_SECRET=your-secret` in `.env`, use same value here
5. Events: select **Pull requests**
6. Save

### 3c. Trigger the webhook
Open or update a PR in your repo. CodeSentinel will:
1. Receive the webhook
2. Verify the HMAC signature
3. Fetch the diff
4. Run security + quality agents in parallel
5. Post a review comment to the PR

---

## Step 4 — Run Tests

```bash
cd codesentinel
source .venv/bin/activate

uv run pytest tests/ -v
```

Expected output:
```
tests/test_diff_parser.py::TestDetectLanguage::test_python PASSED
tests/test_diff_parser.py::TestDetectLanguage::test_typescript PASSED
tests/test_diff_parser.py::TestDetectLanguage::test_dockerfile PASSED
tests/test_diff_parser.py::TestParseDiff::test_parse_empty_diff PASSED
tests/test_diff_parser.py::TestParseDiff::test_parse_single_file PASSED
tests/test_diff_parser.py::TestParseDiff::test_parse_added_lines PASSED
tests/test_diff_parser.py::TestOwaspPatterns::test_detects_sql_injection PASSED
tests/test_diff_parser.py::TestOwaspPatterns::test_detects_hardcoded_secret PASSED
tests/test_diff_parser.py::TestOwaspPatterns::test_clean_code_no_findings PASSED
tests/test_diff_parser.py::TestOwaspPatterns::test_finding_has_required_fields PASSED
tests/test_diff_parser.py::TestSummarizeDiff::test_empty PASSED
tests/test_agents.py::TestGraphCompilation::test_review_graph_compiles PASSED
tests/test_agents.py::TestGraphCompilation::test_graph_has_expected_nodes PASSED
tests/test_agents.py::TestSecurityAgent::test_returns_findings_for_vulnerable_code PASSED
tests/test_agents.py::TestSecurityAgent::test_returns_empty_for_no_diff PASSED
tests/test_agents.py::TestWebhookSignature::test_valid_signature PASSED
tests/test_agents.py::TestWebhookSignature::test_invalid_signature PASSED
tests/test_agents.py::TestWebhookSignature::test_missing_prefix_rejected PASSED
```

---

## Step 5 — View API Docs

Open: **http://localhost:8001/docs**

---

## Teardown

```bash
docker compose -f docker/docker-compose.yml down -v
# This also removes the SQLite file if you used the file-based path inside the container
```

---

## Known Limitations

### 1. Static OWASP patterns have false positives
The 9 regex patterns match based on syntax, not context. For example, `password = "test"` in a test file would trigger the hardcoded-secret rule. The LLM analysis provides context, but the static findings are always included.

### 2. Diff size limit
Large PRs (>8000 characters of diff) are truncated before being sent to the LLM to stay within context window limits. The OWASP static scan still runs on the full diff, but the LLM semantic analysis may miss issues in the truncated portion.

### 3. No streaming — polling only
Results are persisted to SQLite and polled via `GET /review/{id}`. There is no SSE or WebSocket streaming. For real-time feedback, implement a polling loop on the client side.

### 4. Python/JavaScript focus
The OWASP patterns and LLM prompts are optimized for Python and JavaScript. Other languages (Go, Rust, Java) benefit from the LLM analysis but get fewer static pattern hits.

### 5. GitHub PAT scope
The PAT needs `repo` scope to post review comments. Read-only tokens (`public_repo`) work for public repos only. For private repos, full `repo` scope is required.

### 6. No re-review deduplication
If the same PR is reviewed multiple times (e.g., multiple webhook deliveries), each creates a separate review record and posts a separate GitHub comment. There's no deduplication — the same PR could accumulate multiple CodeSentinel reviews.

### 7. Webhook requires public URL
GitHub webhooks need a publicly accessible URL. For local testing, ngrok is required. There's no built-in tunneling.

### 8. SQLite not suitable for concurrent high volume
The review store uses SQLite with `aiosqlite`. SQLite serializes writes, so concurrent reviews are fine but could slow down under very high webhook throughput. Replace with PostgreSQL for production.

### 9. No retry on LLM failure
If the LLM call fails mid-review (timeout, rate limit), the review is marked `failed` and no GitHub comment is posted. There's no automatic retry mechanism.

### 10. Quality analysis is LLM-only (no static tools)
Unlike the security agent (which combines static regex + LLM), the quality agent is purely LLM-based. There's no integration with `pylint`, `eslint`, or similar static analysis tools. Adding those would significantly improve accuracy.
