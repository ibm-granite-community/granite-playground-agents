from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False, extra="allow")

    PORT: int = Field(default=8000, description="HTTP Port the agent will listen on")
    HOST: str = Field(default="127.0.0.1", description="Network address the agent will bind to")
    ACCESS_LOG: bool = Field(default=False, description="Whether the agent logs HTTP access requests")


settings = Settings()
