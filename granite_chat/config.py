import os
from typing import Literal

from pydantic import model_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    port: int = 8000
    host: str = "0.0.0.0"
    ACCESS_LOG: bool = False

    LLM_MODEL: str | None = None
    LLM_API_BASE: str | None = None
    LLM_API_KEY: str | None = None
    LLM_API_HEADERS: str | None = None

    RETRIEVER: str = "google"
    GOOGLE_API_KEY: str | None = None
    GOOGLE_CX_KEY: str | None = None

    OLLAMA_BASE_URL: str = "http://localhost:11434"

    EMBEDDINGS_PROVIDER: str = "watsonx"

    EMBEDDINGS_MODEL: str = "ibm/slate-125m-english-rtrvr-v2"
    EMBEDDINGS_HF_TOKENIZER: str = "FacebookAI/roberta-base"
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 20

    # Populate these vars to enable lora citations via granite-io
    # Otherwise agent will fall back on default implementation
    GRANITE_IO_OPENAI_API_BASE: str | None = None
    GRANITE_IO_CITATIONS_MODEL_ID: str | None = None
    GRANITE_IO_OPENAI_API_HEADERS: str | None = None

    # WATSONX EMBEDDINGS
    # Setting WATSONX_EMBEDDING_MODEL will override default embedding settings
    WATSONX_API_BASE: str | None = None
    WATSONX_PROJECT_ID: str | None = None
    WATSONX_REGION: str | None = None
    WATSONX_API_KEY: str | None = None

    # openai embeddings
    EMBEDDINGS_OPENAI_API_KEY: str | None = None
    EMBEDDINGS_OPENAI_API_BASE: str | None = None
    EMBEDDINGS_OPENAI_API_HEADERS: str | None = None

    MAX_TOKENS: int = 4096
    TEMPERATURE: float = 0.2
    CHAT_TOKEN_LIMIT: int = 10000

    log_level: Literal["FATAL", "ERROR", "WARNING", "INFO", "DEBUG", "TRACE"] = "INFO"

    @model_validator(mode="after")
    def set_secondary_env(self) -> "Settings":
        # We need OLLAMA_BASE_URL to be set in the event that ollama embeddings are used
        if "OLLAMA_BASE_URL" not in os.environ and self.EMBEDDINGS_PROVIDER == "ollama":
            os.environ["OLLAMA_BASE_URL"] = self.OLLAMA_BASE_URL

        # We need RETRIEVER to be set
        if "RETRIEVER" not in os.environ:
            os.environ["RETRIEVER"] = self.RETRIEVER

        return self

    class Config:
        env_file = ".env"


settings = Settings()
