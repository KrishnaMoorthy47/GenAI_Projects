from __future__ import annotations

from functools import lru_cache

from langchain_groq import ChatGroq
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file="../.env", extra="ignore")

    # Application
    api_key: str
    port: int = 8006

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

    # MongoDB
    mongodb_uri: str = "mongodb://localhost:27017"


@lru_cache
def get_settings() -> Settings:
    return Settings()


def get_llm(temperature: float = 0.7):
    s = get_settings()
    if s.llm_provider == "groq":
        return ChatGroq(
            model=s.groq_model,
            api_key=s.groq_api_key,
            temperature=temperature,
        )
    if s.llm_provider == "azure_openai":
        return AzureChatOpenAI(
            azure_endpoint=s.azure_openai_endpoint,
            azure_deployment=s.azure_openai_deployment,
            openai_api_key=s.azure_openai_api_key,
            openai_api_version=s.azure_openai_api_version,
            temperature=temperature,
        )
    return ChatOpenAI(
        model=s.openai_model,
        api_key=s.openai_api_key,
        temperature=temperature,
    )
