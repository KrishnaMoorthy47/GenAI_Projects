"""VoiceAgent configuration via Pydantic Settings."""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file="../.env", extra="ignore")

    # Application
    api_key: str
    port: int = 8004

    # Groq (LLM, STT, TTS) — llama-3.1-8b-instant was retired (Aug 2026);
    # openai/gpt-oss-20b is Groq's own migration recommendation, same fix
    # applied in chatbot. STT/TTS moved off OpenAI onto Groq's own
    # OpenAI-compatible audio endpoints, so this is the only key VoiceAgent
    # needs now.
    groq_api_key: str = ""
    groq_model: str = "openai/gpt-oss-20b"
    tts_voice: str = "autumn"
    tts_model: str = "canopylabs/orpheus-v1-english"


@lru_cache
def get_settings() -> Settings:
    return Settings()
