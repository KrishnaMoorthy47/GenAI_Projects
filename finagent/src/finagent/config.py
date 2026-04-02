from __future__ import annotations

from functools import lru_cache

from langchain_groq import ChatGroq
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file="../.env", extra="ignore")

    # Application
    api_key: str
    port: int = 8000

    # LLM
    llm_provider: str = "groq"  # "groq" | "openai" | "azure_openai"

    # Groq
    groq_api_key: str = ""
    groq_model: str = "llama-3.1-8b-instant"

    # OpenAI
    openai_api_key: str = ""
    openai_model: str = "gpt-4o"

    # Azure OpenAI
    azure_openai_endpoint: str = ""
    azure_openai_deployment: str = "gpt-4o"
    azure_openai_api_key: str = ""
    azure_openai_api_version: str = "2024-08-01-preview"

    # Tools
    tavily_api_key: str = ""

    # Database (Supabase or any Postgres)
    # Format: postgresql://postgres:[PASSWORD]@db.[PROJECT].supabase.co:5432/postgres?sslmode=require
    database_url: str = ""

    # Local testing
    use_memory_saver: bool = False


@lru_cache
def get_settings() -> Settings:
    return Settings()


def get_llm(streaming: bool = False):
    settings = get_settings()
    if settings.llm_provider == "groq":
        return ChatGroq(
            model=settings.groq_model,
            api_key=settings.groq_api_key,
            streaming=streaming,
            temperature=0,
        )
    # if settings.llm_provider == "azure_openai":
    #     return AzureChatOpenAI(
    #         azure_endpoint=settings.azure_openai_endpoint,
    #         azure_deployment=settings.azure_openai_deployment,
    #         openai_api_key=settings.azure_openai_api_key,
    #         openai_api_version=settings.azure_openai_api_version,
    #         streaming=streaming,
    #         temperature=0,
    #     )
    return ChatOpenAI(
        model=settings.openai_model,
        api_key=settings.openai_api_key,
        streaming=streaming,
        temperature=0,
    )
