# VoiceAgent — Architecture

A real-time voice AI agent over WebSocket. The client streams raw PCM audio — the server transcribes with Whisper, generates a response with Groq LLM (sentence-streamed), synthesizes each sentence to audio with OpenAI TTS, and sends base64-encoded MP3 chunks back. A static browser frontend is included for testing.

## Flow

```
Browser / Client
    │  (raw PCM bytes — 16kHz, 16-bit, mono)
    ▼
FastAPI  (port 8004)
    │
    ▼  WS /ws/voice/{session_id}
┌─────────────────────────────────┐
│  Voice Pipeline (per message)   │
│                                 │
│  PCM bytes → Whisper STT        │
│       → Groq LLM (streaming)    │
│       → OpenAI TTS per sentence │
│       → base64 MP3 → client     │
└─────────────────────────────────┘
    │
SessionStore  (in-memory, keyed by session UUID)
```

## API

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| GET | `/health` | No | Health check |
| POST | `/api/v1/session` | Yes | Create a new voice session |
| DELETE | `/api/v1/session/{id}` | Yes | Delete a session |
| WS | `/ws/voice/{session_id}` | No | Real-time voice pipeline |
| GET | `/` | No | Browser test frontend |

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `API_KEY` | required | `x-api-key` header value |
| `OPENAI_API_KEY` | — | Used for Whisper STT + TTS |
| `TTS_VOICE` | `alloy` | OpenAI TTS voice |
| `TTS_MODEL` | `tts-1` | OpenAI TTS model |
| `GROQ_API_KEY` | — | Used for LLM streaming |
| `GROQ_MODEL` | `llama-3.1-8b-instant` | Groq model |
| `PORT` | `8004` | Server port |
