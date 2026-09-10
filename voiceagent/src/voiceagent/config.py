"""VoiceAgent configuration via Pydantic Settings."""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file="../.env", extra="ignore")

    # Application
    api_key: str
    port: int = 8004

    # OpenAI (Whisper STT + TTS)
    openai_api_key: str = ""
    tts_voice: str = "alloy"
    tts_model: str = "tts-1"

    # Groq (LLM) — llama-3.1-8b-instant was retired (Aug 2026); openai/gpt-oss-20b
    # is Groq's own migration recommendation, same fix applied in chatbot.
    groq_api_key: str = ""
    groq_model: str = "openai/gpt-oss-20b"


@lru_cache
def get_settings() -> Settings:
    return Settings()
