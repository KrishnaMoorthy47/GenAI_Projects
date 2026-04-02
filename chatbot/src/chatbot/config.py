from __future__ import annotations

import json
from functools import lru_cache
from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file="../.env", extra="ignore")

    # Application
    app_env: str = "development"
    app_version: str = "1.0.0"
    app_port: int = 8002

    # Authentication
    api_key: str

    # LLM
    llm_provider: str = "groq"  # "groq" | "openai" | "ollama" | "azure_openai"
    llm_model: str = "gpt-4o"
    llm_max_tokens: int = 1500
    llm_temperature: float = 0.2

    # Groq
    groq_api_key: str = ""
    groq_model: str = "llama-3.1-8b-instant"

    # OpenAI (also used for embeddings regardless of llm_provider)
    openai_api_key: str = ""
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536

    # Ollama
    ollama_base_url: str = "http://localhost:11434"

    # Azure OpenAI (used when llm_provider="azure_openai")
    azure_openai_endpoint: str = ""
    azure_openai_deployment: str = ""
    azure_openai_api_key: str = ""
    azure_openai_api_version: str = "2024-08-01-preview"
    azure_openai_embedding_deployment: str = ""

    # Vector Store
    data_input_dir: str = "data/input"
    data_indexed_dir: str = "data/indexed"
    default_collection: str = "default"
    vector_search_k: int = 5

    # RAG Pipeline
    max_context_tokens: int = 4000
    threshold_ratio: float = 0.8
    max_query_length: int = 1000
    enable_relevance_precheck: bool = False

    # Chat History
    max_history_turns: int = 5
    cache_ttl: int = 3600

    # Redis
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_password: str = ""
    redis_db: int = 0

    # Ingestion
    chunk_size: int = 575
    chunk_overlap: int = 400

    # Content Safety
    enable_content_safety: bool = True

    # Supabase pgvector
    supabase_db_url: str = ""
    vector_backend: str = "supabase"  # "supabase" | "faiss"

    # Groq vision
    groq_vision_model: str = "llama-3.2-11b-vision-preview"
    vision_text_threshold: int = 50  # chars per PDF page below which vision is used

    # CORS (stored as JSON string, parsed below)
    cors_origins: str = '["http://localhost:3000"]'

    def get_cors_origins(self) -> List[str]:
        try:
            return json.loads(self.cors_origins)
        except (json.JSONDecodeError, TypeError):
            return ["http://localhost:3000"]


@lru_cache
def get_settings() -> Settings:
    return Settings()
