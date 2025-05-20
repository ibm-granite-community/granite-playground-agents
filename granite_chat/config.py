from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    port: int = 8000
    host: str = "0.0.0.0"
    LLM_MODEL: str = None
    LLM_API_BASE: str = None
    LLM_API_KEY: str = None
    LLM_API_HEADERS: str | None = None
    max_tokens: int = 4096
    temperature: float = 0.3

    class Config:
        env_file = ".env"


settings = Settings()
