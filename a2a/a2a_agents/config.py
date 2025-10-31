# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0


from agentstack_sdk.a2a.extensions import AgentDetail, AgentDetailContributor
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False, extra="allow")

    PORT: int = Field(default=8000, description="HTTP Port the agent will listen on")
    HOST: str = Field(default="127.0.0.1", description="Network address the agent will bind to")
    ACCESS_LOG: bool = Field(default=False, description="Whether the agent logs HTTP access requests")


agent_detail = AgentDetail(
    interaction_mode="multi-turn",
    user_greeting="Hi, I'm Granite! How can I help you?",
    framework="BeeAI",
    license="Apache 2.0",
    programming_language="Python",
    homepage_url="https://github.ibm.com/research-design-tech-experiences/beeai-platform-granite-chat/",
    source_code_url="https://github.ibm.com/research-design-tech-experiences/beeai-platform-granite-chat/",
    author=AgentDetailContributor(name="IBM Research", url="https://www.ibm.com"),
)

settings = Settings()
