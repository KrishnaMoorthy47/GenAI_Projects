"""Configuration and LLM factory for agenteval."""

import logging
from functools import lru_cache
from typing import Literal

from langchain_groq import ChatGroq
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file="../.env", extra="ignore")

    api_key: str
    llm_provider: Literal["groq", "openai", "azure_openai"] = "groq"

    # Groq
    groq_api_key: str = ""
    groq_model: str = "llama-3.1-8b-instant"

    # OpenAI
    openai_api_key: str = ""
    openai_model: str = "gpt-4o"

    # Azure OpenAI (only needed when llm_provider=azure_openai)
    azure_openai_api_key: str = ""
    azure_openai_endpoint: str = ""
    azure_openai_api_version: str = "2024-08-01-preview"
    azure_openai_deployment: str = "gpt-4o"

    langsmith_api_key: str = ""
    langsmith_project: str = "agenteval"
    langsmith_tracing: bool = False

    port: int = 8003
    db_path: str = "data/tamil_songs.db"
    max_agent_iterations: int = 8


@lru_cache
def get_settings() -> Settings:
    return Settings()


def get_llm(temperature: float = 0.0):
    """Return a ChatGroq, ChatOpenAI, or AzureChatOpenAI instance based on LLM_PROVIDER."""
    settings = get_settings()

    if settings.llm_provider == "groq":
        logger.info("Using ChatGroq: %s", settings.groq_model)
        return ChatGroq(
            model=settings.groq_model,
            api_key=settings.groq_api_key,
            temperature=temperature,
        )

    # if settings.llm_provider == "azure_openai":
    #     logger.info("Using AzureChatOpenAI: %s", settings.azure_openai_deployment)
    #     return AzureChatOpenAI(
    #         azure_endpoint=settings.azure_openai_endpoint,
    #         azure_deployment=settings.azure_openai_deployment,
    #         openai_api_version=settings.azure_openai_api_version,
    #         api_key=settings.azure_openai_api_key,
    #         temperature=temperature,
    #     )

    logger.info("Using ChatOpenAI: %s", settings.openai_model)
    return ChatOpenAI(
        model=settings.openai_model,
        api_key=settings.openai_api_key or None,
        temperature=temperature,
    )
