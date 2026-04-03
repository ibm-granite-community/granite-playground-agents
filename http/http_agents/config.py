# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0


from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False, extra="allow")

    # API Server
    PORT: int = Field(default=8000, description="HTTP Port the API will listen on")
    HOST: str = Field(default="127.0.0.1", description="Network address the API will bind to")
    API_PREFIX: str = Field(default="/api/v1", description="API route prefix")

    # CORS
    CORS_ORIGINS: list[str] = Field(default=["*"], description="Allowed CORS origins")

    # Session Management
    SESSION_TTL_HOURS: int = Field(default=24, description="Session time-to-live in hours")
    MAX_HISTORY_MESSAGES: int = Field(default=100, description="Maximum messages to keep in history")

    # Streaming
    HEARTBEAT_INTERVAL: float = Field(default=10, description="Interval between heartbeat messages in seconds")

    # Debug and logging
    DEBUG: bool = Field(default=False, description="Enable debug mode")
    LOG_LEVEL: str = Field(default="INFO", description="Logging level")

    # Key store (Redis for session persistence)
    KEY_STORE_PROVIDER: Literal["redis", "memory"] = Field(default="memory", description="Session storage backend")
    REDIS_CLIENT_URL: SecretStr | None = Field(default=None, description="Redis client URL for session persistence")

    # Resource store (S3 for file storage)
    RESOURCE_STORE_PROVIDER: Literal["S3"] | None = None
    S3_BUCKET: str | None = Field(default=None, description="S3 bucket name")
    S3_ENDPOINT: str | None = Field(default=None, description="S3 resource store endpoint")
    S3_ACCESS_KEY_ID: SecretStr | None = Field(default=None, description="S3 access key id")
    S3_SECRET_ACCESS_KEY: SecretStr | None = Field(default=None, description="S3 secret access key")


settings = Settings()
