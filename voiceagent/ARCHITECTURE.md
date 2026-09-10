# VoiceAgent — Architecture

A real-time voice AI agent over WebSocket. The client streams raw PCM audio — the server transcribes with Groq's Whisper, generates a response with Groq LLM (sentence-streamed), synthesizes each sentence to audio with Groq's Orpheus TTS, and sends base64-encoded WAV chunks back. A static browser frontend is included for testing.

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
│  PCM bytes → Groq Whisper STT   │
│       → Groq LLM (streaming)    │
│       → Groq Orpheus per sentence│
│       → base64 WAV → client     │
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
| `GROQ_API_KEY` | — | Used for LLM streaming, Whisper STT, and Orpheus TTS |
| `TTS_VOICE` | `autumn` | Groq Orpheus TTS voice |
| `TTS_MODEL` | `canopylabs/orpheus-v1-english` | Groq Orpheus TTS model |
| `GROQ_MODEL` | `openai/gpt-oss-20b` | Groq model |
| `PORT` | `8004` | Server port |
