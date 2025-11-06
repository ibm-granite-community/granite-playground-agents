# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0


from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False, extra="allow")

    # Agent server
    PORT: int = Field(default=8000, description="HTTP Port the agent will listen on")
    HOST: str = Field(default="127.0.0.1", description="Network address the agent will bind to")
    ACCESS_LOG: bool = Field(default=False, description="Whether the agent logs HTTP access requests")

    # Agent behaviour
    TWO_STEP_THINKING: bool = Field(default=False, description="Enable two step thinking.")
    HEARTBEAT_INTERVAL: float = Field(default=10, description="Interval between heartbeat messages.")

    # Key store
    KEY_STORE_PROVIDER: Literal["redis"] | None = None
    REDIS_CLIENT_URL: SecretStr | None = Field(
        default=None, description="Redis client object configured from the given URL"
    )

    # Resource store
    RESOURCE_STORE_PROVIDER: Literal["S3"] | None = None
    S3_BUCKET: str | None = Field(default=None, description="S3 bucket name")
    S3_ENDPOINT: str | None = Field(default=None, description="S3 resource store endpoint")
    S3_ACCESS_KEY_ID: SecretStr | None = Field(default=None, description="S3 access key id")
    S3_SECRET_ACCESS_KEY: SecretStr | None = Field(default=None, description="S3 secret access ket")

    # Memory store
    MEM_STORE_NOTIFICATION_DEBOUNCE: float = Field(
        default=0.2, description="Memory store notification debounce (seconds)"
    )


settings = Settings()
