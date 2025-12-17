# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0


from collections.abc import AsyncGenerator
from typing import Annotated

from a2a.types import Message as A2AMessage
from agentstack_sdk.a2a.extensions import (
    CitationExtensionServer,
    CitationExtensionSpec,
    EmbeddingServiceExtensionServer,
    EmbeddingServiceExtensionSpec,
    LLMServiceExtensionServer,
    LLMServiceExtensionSpec,
    OptionItem,
    SettingsExtensionServer,
    SettingsExtensionSpec,
    SettingsRender,
    SingleSelectField,
    SingleSelectFieldValue,
    TrajectoryExtensionServer,
    TrajectoryExtensionSpec,
)
from agentstack_sdk.a2a.types import RunYield
from agentstack_sdk.server import Server
from agentstack_sdk.server.context import RunContext
from agentstack_sdk.server.store.platform_context_store import PlatformContextStore
from granite_core.config import settings as core_settings
from granite_core.logging import get_logger
from granite_core.utils import log_settings

from a2a_agents import __version__
from a2a_agents.agents.chat.agent_chat import chat, chat_skill
from a2a_agents.agents.research.agent_research import research, research_skill
from a2a_agents.agents.search.agent_search import search, search_skill
from a2a_agents.config import agent_detail, settings

logger = get_logger(__name__)
server = Server()


if settings.USE_AGENTSTACK_LLM:
    # agent with LLM extensions
    @server.agent(
        name="Granite Playground",
        description="This agent leverages the IBM Granite models for general chat, search and research.",
        version=__version__,
        detail=agent_detail,
        skills=[chat_skill, search_skill, research_skill],
    )
    async def agent(
        input: A2AMessage,
        context: RunContext,
        llm_ext: Annotated[
            LLMServiceExtensionServer,
            LLMServiceExtensionSpec.single_demand(suggested=(settings.SUGGESTED_LLM_MODEL,)),
        ],
        embedding_ext: Annotated[
            EmbeddingServiceExtensionServer,
            EmbeddingServiceExtensionSpec.single_demand(suggested=(settings.SUGGETED_EMBEDDING_MODEL,)),
        ],
        trajectory: Annotated[TrajectoryExtensionServer, TrajectoryExtensionSpec()],
        citation: Annotated[CitationExtensionServer, CitationExtensionSpec()],
        settings: Annotated[
            SettingsExtensionServer,
            SettingsExtensionSpec(
                params=SettingsRender(
                    fields=[
                        SingleSelectField(
                            id="agent_type",
                            label="type",
                            options=[
                                OptionItem(value="chat", label="Chat"),
                                OptionItem(value="search", label="Search"),
                                OptionItem(value="research", label="Research"),
                            ],
                            default_value="chat",
                        )
                    ],
                ),
            ),
        ],
    ) -> AsyncGenerator[RunYield, A2AMessage]:
        # parse options
        agent_type = "chat"
        if settings:
            agent_type_settings = settings.parse_settings_response().values["agent_type"]
            if isinstance(agent_type_settings, SingleSelectFieldValue):
                agent_type = agent_type_settings.value if agent_type_settings.value else "chat"
        logger.info(f"Running {agent_type} agent")

        # run the agent
        match agent_type:
            case "chat":
                async for response in chat(input, context, llm_ext):
                    yield response
            case "search":
                async for response in search(input, context, trajectory, citation, llm_ext, embedding_ext):
                    yield response
            case "research":
                async for response in research(input, context, trajectory, citation, llm_ext, embedding_ext):
                    yield response

else:
    # agent without LLM extensions
    @server.agent(
        name="Granite Playground",
        description="This agent leverages the IBM Granite models for general chat, search and research.",
        version=__version__,
        detail=agent_detail,
        skills=[chat_skill, search_skill, research_skill],
    )
    async def agent(
        input: A2AMessage,
        context: RunContext,
        trajectory: Annotated[TrajectoryExtensionServer, TrajectoryExtensionSpec()],
        citation: Annotated[CitationExtensionServer, CitationExtensionSpec()],
        settings: Annotated[
            SettingsExtensionServer,
            SettingsExtensionSpec(
                params=SettingsRender(
                    fields=[
                        SingleSelectField(
                            id="agent_type",
                            label="type",
                            options=[
                                OptionItem(value="chat", label="Chat"),
                                OptionItem(value="search", label="Search"),
                                OptionItem(value="research", label="Research"),
                            ],
                            default_value="chat",
                        )
                    ],
                ),
            ),
        ],
    ) -> AsyncGenerator[RunYield, A2AMessage]:
        # parse options
        agent_type = "chat"
        if settings:
            agent_type_settings = settings.parse_settings_response().values["agent_type"]
            if isinstance(agent_type_settings, SingleSelectFieldValue):
                agent_type = agent_type_settings.value if agent_type_settings.value else "chat"
        logger.info(f"Running {agent_type} agent")

        # run the agent
        match agent_type:
            case "chat":
                async for response in chat(input, context):
                    yield response
            case "search":
                async for response in search(input, context, trajectory, citation):
                    yield response
            case "research":
                async for response in research(input, context, trajectory, citation):
                    yield response


if __name__ == "__main__":
    log_settings(settings, name="Agent")
    log_settings(core_settings)
    server.run(
        host=settings.HOST,
        port=settings.PORT,
        access_log=settings.ACCESS_LOG,
        context_store=PlatformContextStore(),
    )
