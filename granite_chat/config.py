from typing import Literal

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    agent_name: str = "Granite Chat"

    port: int = 8000
    host: str = "0.0.0.0"

    LLM_MODEL: str | None = None
    LLM_API_BASE: str | None = None
    LLM_API_KEY: str | None = None
    LLM_API_HEADERS: str | None = None

    RETRIEVER: str | None = None
    GOOGLE_API_KEY: str | None = None
    GOOGLE_CX_KEY: str | None = None

    OLLAMA_BASE_URL: str | None = None
    EMBEDDING: str | None = None

    # Populate these vars to enable lora citations via granite-io
    # Otherwise agent will fall back on default implementation
    GRANITE_IO_OPENAI_API_BASE: str | None = None
    GRANITE_IO_CITATIONS_MODEL_ID: str | None = None
    GRANITE_IO_OPENAI_API_HEADERS: str | None = None

    max_tokens: int = 4096
    temperature: float = 0.2
    search: bool = True
    thinking: bool = False

    log_level: Literal["FATAL", "ERROR", "WARNING", "INFO", "DEBUG", "TRACE"] = "INFO"

    class Config:
        env_file = ".env"


settings = Settings()
