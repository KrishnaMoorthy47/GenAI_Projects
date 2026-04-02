# Copyright (c) 2024 ValGenesis Inc. All rights reserved.
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

    # Groq (LLM)
    groq_api_key: str = ""
    groq_model: str = "llama-3.1-8b-instant"


@lru_cache
def get_settings() -> Settings:
    return Settings()
