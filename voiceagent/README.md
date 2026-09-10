# VoiceAgent — Real-Time Voice AI over WebSocket

A portfolio project demonstrating a **real-time voice AI pipeline**: raw PCM audio in, synthesized speech out — fully streamed over a single WebSocket connection.

---

## What's Inside

| Component | Description |
|-----------|-------------|
| **STT** | Groq Whisper transcribes raw PCM audio to text |
| **LLM** | Groq LLaMA streams a response, sentence by sentence |
| **TTS** | Groq Orpheus converts each sentence to WAV audio |
| **WebSocket** | Bidirectional pipeline: PCM bytes in, base64 WAV chunks out |
| **Frontend** | Static browser test client mounted at `/` |

---

## Architecture

```
Browser / Client
    │  (raw PCM bytes — 16kHz, 16-bit, mono)
    ▼
FastAPI  (port 8004)
    │
    ▼  WS /ws/voice/{session_id}
┌─────────────────────────────────────────────┐
│  Voice Pipeline (per message)               │
│                                             │
│  PCM bytes → Groq Whisper STT → transcript  │
│       → Groq LLaMA (sentence streaming)     │
│       → Groq Orpheus TTS per sentence       │
│       → {"type":"audio","data":"<base64>"}  │
│       → {"type":"done"}                     │
└─────────────────────────────────────────────┘
    │
SessionStore  (in-memory, keyed by session UUID)
```

---

## Quickstart

```bash
# 1. Install dependencies
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env
# Edit .env — set GROQ_API_KEY, API_KEY

# 3. Run server
uv run uvicorn voiceagent.main:app --port 8004 --reload

# 4. Open the browser test client
# Navigate to http://localhost:8004 and click "Start Recording"

# 5. Run tests
uv run pytest tests/ -v
```

---

## API

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `GET` | `/health` | No | Health check |
| `POST` | `/api/v1/session` | Yes | Create a new voice session |
| `DELETE` | `/api/v1/session/{id}` | Yes | Delete a session |
| `WS` | `/ws/voice/{session_id}` | No | Real-time voice pipeline |
| `GET` | `/` | No | Browser test frontend |

**WebSocket message format:**
- Send: raw PCM bytes (16kHz, 16-bit, mono)
- Receive: `{"type": "transcript", "text": "..."}` → `{"type": "audio", "data": "<base64 WAV>"}` (one per sentence) → `{"type": "done"}`

---

## Project Structure

```
voiceagent/
├── src/voiceagent/
│   ├── main.py          # FastAPI app (lifespan, auth, WebSocket router)
│   ├── config.py        # Settings (pydantic-settings)
│   ├── models.py        # AudioMessage, TranscriptMessage, DoneMessage
│   ├── services/
│   │   ├── stt.py       # Whisper transcription (Groq)
│   │   ├── llm.py       # Groq LLaMA sentence streaming
│   │   ├── tts.py       # Groq Orpheus TTS → WAV bytes
│   │   └── session.py   # In-memory session store
│   └── static/          # Browser test frontend (HTML/JS)
├── tests/               # 4 test modules (all mocked)
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
└── .env.example
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `API_KEY` | required | `x-api-key` header auth |
| `GROQ_API_KEY` | — | LLM streaming, Whisper STT, and Orpheus TTS |
| `TTS_VOICE` | `autumn` | Groq Orpheus TTS voice |
| `TTS_MODEL` | `canopylabs/orpheus-v1-english` | Groq Orpheus TTS model |
| `GROQ_MODEL` | `openai/gpt-oss-20b` | Groq model |
| `PORT` | `8004` | Server port |

---

## Docker

```bash
docker-compose -f docker/docker-compose.yml up --build
```

> Run from within the `voiceagent/` directory. The build context extends to the repo root to include the shared `rate_limit.py` middleware.

---

## Key Design Decisions

- **Sentence-level streaming** — LLM output is buffered until `.!?` punctuation, then TTS fires per sentence; reduces perceived latency vs waiting for the full response
- **Sessions in-memory** — no database required; sessions are keyed by UUID and scoped to a single server instance
- **Groq for STT/TTS** — Whisper and Orpheus via Groq's OpenAI-compatible audio endpoints keep the service stateless, avoid a GPU dependency, and mean the whole service runs on a single API key
