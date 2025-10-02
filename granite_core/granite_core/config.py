import os
from typing import Annotated, Literal

from pydantic import AfterValidator, Field, SecretStr, TypeAdapter, model_validator
from pydantic.networks import HttpUrl
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False, extra="allow")

    STREAMING: bool = Field(default=True, description="Stream user facing content")

    LLM_PROVIDER: Literal["openai", "watsonx"] = "openai"
    LLM_MODEL: str | None = Field(description="The model ID of the LLM")
    LLM_STRUCTURED_MODEL: str | None = Field(
        description="The model ID of the LLM used for structured generation tasks", default=None
    )

    LLM_API_BASE: Annotated[
        HttpUrl | None, Field(description="The OpenAI base URL for chat completions"), AfterValidator(str)
    ] = None
    LLM_API_KEY: SecretStr | None = Field(
        description="The authorization key used to access LLM_MODEL via LLM_API_BASE", default=None
    )
    LLM_API_HEADERS: SecretStr | None = Field(description="Additional headers to provide to LLM_API_BASE", default=None)
    LLM_TIMEOUT: float = Field(description="Timeout for llm generation requests", default=180)
    MAX_RETRIES: int = Field(description="Max retries for inference", default=3)

    RETRIEVER: Literal["google", "tavily"] = Field(default="google", description="The search engine to use")
    GOOGLE_API_KEY: SecretStr | None = Field(description="The API key for Google Search")
    GOOGLE_CX_KEY: SecretStr | None = Field(description="The CX key for Google Search")
    TAVILY_API_KEY: SecretStr | None = Field(default=None, description="The API key for Tavily")
    SAFE_SEARCH: bool = Field(default=True, description="Turn on safe search if available for search engine.")

    SCRAPER_MAX_CONTENT_LENGTH: int = Field(
        description="Max size of scraped content in characters, anything larger will be truncated.", default=15000
    )

    SCRAPER_TIMEOUT: int = Field(description="Seconds elapsed before scraper task times out.", default=15)

    OLLAMA_BASE_URL: Annotated[
        HttpUrl,
        Field(
            default_factory=lambda: TypeAdapter(HttpUrl).validate_python("http://localhost:11434"),
            description="The OpenAI base URL for chat completions",
        ),
        AfterValidator(str),
    ]

    EMBEDDINGS_PROVIDER: Literal["watsonx", "ollama", "openai"] = Field(
        default="watsonx", description="Which provider to use for calculating embeddings"
    )

    EMBEDDINGS_MODEL: str = Field(
        default="ibm/slate-125m-english-rtrvr-v2", description="The model ID of the embedding model"
    )
    EMBEDDINGS_HF_TOKENIZER: str = Field(default="FacebookAI/roberta-base", description="The model ID of the tokenizer")
    EMBEDDINGS_MAX_SEQUENCE: int = Field(default=512, description="The maximum sequence length in tokens.")

    CHUNK_SIZE: int = Field(
        default=512,
        description="The maximum number of characters (or tokens if configured) search result chunks are broken into",
    )
    CHUNK_OVERLAP: int = Field(
        default=20, description="The number of characters or tokens search result chunks will overlap"
    )

    MAX_EMBEDDINGS_PER_REQUEST: int = Field(default=200, description="The max number of embeddings in a single request")

    EMBEDDINGS_SIM_MODEL: str | None = Field(
        default=None, description="The model ID of the embedding model used for similarity."
    )
    EMBEDDINGS_SIM_HF_TOKENIZER: str = Field(
        default="FacebookAI/xlm-roberta-base", description="The model ID of the tokenizer"
    )
    EMBEDDINGS_SIM_MAX_SEQUENCE: int = Field(default=512, description="The maximum sequence length in tokens.")

    # Populate these vars to enable lora citations via granite-io
    # Otherwise agent will fall back on default implementation
    GRANITE_IO_OPENAI_API_BASE: Annotated[
        HttpUrl | None, Field(default=None, description="The OpenAI base URL for chat completions"), AfterValidator(str)
    ]
    GRANITE_IO_CITATIONS_MODEL_ID: str | None = Field(default=None, description="The model ID for citations")
    GRANITE_IO_OPENAI_API_HEADERS: SecretStr | None = Field(
        default=None, description="Additional headers to provide to GRANITE_IO_OPENAI_API_BASE"
    )

    # WATSONX EMBEDDINGS
    # Setting WATSONX_EMBEDDING_MODEL will override default embedding settings
    WATSONX_API_BASE: Annotated[
        HttpUrl | None, Field(default=None, description="The OpenAI base URL for chat completions"), AfterValidator(str)
    ]
    WATSONX_PROJECT_ID: SecretStr | None = Field(default=None, description="The project ID of your Watsonx deployment")
    WATSONX_REGION: str | None = Field(default=None, description="The region of your Watsonx deployment")
    WATSONX_API_KEY: SecretStr | None = Field(
        default=None, description="The Cloud API Key to reach your Watsonx deployment"
    )

    # openai embeddings
    EMBEDDINGS_OPENAI_API_BASE: Annotated[
        HttpUrl | None, Field(default=None, description="The OpenAI base URL for chat completions"), AfterValidator(str)
    ]
    EMBEDDINGS_OPENAI_API_KEY: SecretStr | None = Field(
        default=None, description="The API key to reach your OpenAI endpoint"
    )
    EMBEDDINGS_OPENAI_API_HEADERS: SecretStr | None = Field(
        default=None, description="Additional headers to provide to EMBEDDINGS_OPENAI_API_BASE"
    )

    MAX_TOKENS: int = Field(
        default=4096, description="The maximum number of tokens the LLM will generate", ge=10, le=128_000
    )

    TEMPERATURE: float = Field(
        default=0,
        description="How predictable (low value) or creative (high value) the LLM responses are",
        ge=0.0,
        le=2.0,
    )

    CHAT_TOKEN_LIMIT: int = Field(
        default=5_000,
        description="The number of tokens that are generated by the LLM before the agent sends a message",
    )

    log_level: Literal["FATAL", "ERROR", "WARNING", "INFO", "DEBUG", "TRACE"] = Field(
        default="INFO", description="Set the log level for the agent"
    )

    SEARCH_MAX_SEARCH_QUERIES_PER_STEP: int = Field(
        default=3, description="Max search queries generated per search run"
    )

    # Search configuration
    SEARCH_MAX_SEARCH_RESULTS_PER_STEP: int = Field(
        default=6, description="Controls how man search results are considered for each search query", ge=1
    )

    SEARCH_MAX_DOCS_PER_STEP: int = Field(
        default=10, description="The number of documents to return from the vector store"
    )
    SEARCH_MAX_SCRAPED_CONTENT: int = Field(default=10, description="The max scraped web results")

    # Research configuration
    RESEARCH_PLAN_BREADTH: int = Field(default=5, description="Controls how many search queries are executed", ge=1)
    RESEARCH_MAX_SEARCH_RESULTS_PER_STEP: int = Field(
        default=8, description="Controls how man search results are considered for each search query", ge=1
    )
    RESEARCH_MAX_DOCS_PER_STEP: int = Field(
        default=10, description="The number of documents to return from the vector store"
    )

    RESEARCH_MAX_SCRAPED_CONTENT: int = Field(default=10, description="The max scraped web results")
    RESEARCH_PRELIM_MAX_TOKENS: int = Field(default=2048, description="Token budget for preliminary research step")
    RESEARCH_FINDINGS_MAX_TOKENS: int = Field(default=2048, description="Token budget for finding research step")

    # Inference throttle
    MAX_CONCURRENT_INFERENCE_TASKS: int = Field(
        default=20,
        description="The max. number of inference operations that can run simultaneously, includes chat and embeddings",
    )
    RATE_LIMIT_INFERENCE_TASKS: int = Field(
        default=8, description="Rate limit for inference tasks in specified rate period"
    )
    RATE_PERIOD_INFERENCE_TASKS: int = Field(
        default=2, description="Rate period in seconds, use with rate limit to implement throttle"
    )

    # General task throttle
    MAX_CONCURRENT_TASKS: int = Field(
        default=30,
        description="The max. number of tasks that can run simultaneously, excludes inference tasks",
    )
    RATE_LIMIT_TASKS: int = Field(
        default=20,
        description="Rate limit for tasks in specified rate period",
    )
    RATE_PERIOD_TASKS: int = Field(
        default=2, description="Rate period in seconds, use with rate limit to implement throttle"
    )

    # Citations
    CITATIONS_MAX_STATEMENTS: int = Field(
        default=10,
        description="The max. number of source statements per response statement",
    )
    CITATIONS_SIM_THRESHOLD: float = Field(
        default=0.8,
        description="The similarity threshold under which citation statements are ignored.",
    )

    # MMR
    MMR_LAMBDA_MULT: float = Field(
        default=0.4,
        description="Controls the weighting between relevance and diversity in MMR",
    )

    MEMORY_STORE_TTL_MINS: int = Field(default=20, description="Memory store key timeout")

    @model_validator(mode="after")
    def set_secondary_env(self) -> "Settings":
        # We need OLLAMA_BASE_URL to be set in the event that ollama embeddings are used
        if "OLLAMA_BASE_URL" not in os.environ and self.EMBEDDINGS_PROVIDER == "ollama":
            os.environ["OLLAMA_BASE_URL"] = str(self.OLLAMA_BASE_URL)

        # We need RETRIEVER to be set
        if "RETRIEVER" not in os.environ:
            os.environ["RETRIEVER"] = self.RETRIEVER

        if self.RETRIEVER == "google" and (self.GOOGLE_API_KEY is None or self.GOOGLE_CX_KEY is None):
            raise ValueError("Google retriever requires GOOGLE_API_KEY and GOOGLE_CX_KEY")
        elif self.RETRIEVER == "tavily" and self.TAVILY_API_KEY is None:
            raise ValueError("Tavily retriever requires TAVILY_API_KEY")

        # Allows headers to be picked up by framework
        if self.LLM_API_HEADERS:
            os.environ["OPENAI_API_HEADERS"] = self.LLM_API_HEADERS.get_secret_value()

        return self


settings = Settings()  # type: ignore[call-arg]
