from collections.abc import AsyncGenerator
from typing import Annotated

from a2a.types import AgentSkill
from a2a.types import Message as A2AMessage
from beeai_sdk.a2a.extensions import (
    AgentDetail,
    AgentDetailContributor,
    CitationExtensionServer,
    CitationExtensionSpec,
    OptionItem,
    SettingsExtensionServer,
    SettingsExtensionSpec,
    SettingsRender,
    SingleSelectField,
    SingleSelectFieldValue,
    TrajectoryExtensionServer,
    TrajectoryExtensionSpec,
)
from beeai_sdk.a2a.types import RunYield
from beeai_sdk.server import Server
from beeai_sdk.server.context import RunContext
from beeai_sdk.server.store.platform_context_store import PlatformContextStore
from granite_core.config import settings as core_settings
from granite_core.logging import get_logger
from granite_core.utils import log_settings

from a2a_agents import __version__
from a2a_agents.agent_chat import chat
from a2a_agents.agent_research import research
from a2a_agents.agent_search import search
from a2a_agents.config import settings

logger = get_logger(__name__)
server = Server()


@server.agent(
    name="Granite Playground",
    description="This agent leverages the IBM Granite models for general chat, search and research.",
    url="https://github.ibm.com/research-design-tech-experiences/beeai-platform-granite-chat/",
    documentation_url="https://github.ibm.com/research-design-tech-experiences/beeai-platform-granite-chat/",
    version=__version__,
    detail=AgentDetail(
        interaction_mode="multi-turn",
        user_greeting="Hi, I'm Granite! How can I help you?",
        framework="BeeAI",
        license="Apache 2.0",
        programming_language="Python",
        homepage_url="https://github.ibm.com/research-design-tech-experiences/beeai-platform-granite-chat/",
        source_code_url="https://github.ibm.com/research-design-tech-experiences/beeai-platform-granite-chat/",
        author=AgentDetailContributor(name="IBM Research", url="https://www.ibm.com"),
    ),
    skills=[
        AgentSkill(
            id="chat",
            name="Chat",
            description="Chat with the model with no external influence",
            tags=["chat"],
            examples=[
                "Explain how RSUs work. I just got offered RSUs in a job offer — what exactly are they and how do they pay out? Break down how RSUs work, including how they vest, the tax implications at each stage, and how I might think about their long-term value compared to salary or stock options."  # noqa: E501
                "Write a thank you note to my colleague Elaine who is celebrating her 10-year work anniversary. Make it short but mention her important contributions to the Granite project and her valuable mentorship of new team members."  # noqa: E501
                "Help me ideate compelling and distinctive project names for my research work on autonomous web agents. The names should feel futuristic but grounded, relevant to AI or web infrastructure, and easy to remember. Provide 5 options with short rationale for each."  # noqa: E501
            ],
        ),
        AgentSkill(
            id="search",
            name="Search",
            description="Chat with the model that's enabled with Internet connected search",
            tags=["chat", "search"],
            examples=[
                "What are the latest innovations in long term memory for LLMs?"
                "What is the LLM usage policy at NeurIPS?"
                "What are the best cities to work remotely from for a month — with reliable Wi-Fi, affordable rentals, reliable infrastructure, and good coffee shops or co-working spaces? Focus on destinations in Europe."  # noqa: E501
            ],
        ),
        AgentSkill(
            id="research",
            name="Research",
            description="Chat with the model that's enabled with Internet connected deep research",
            tags=["research"],
            examples=[
                "Explain the nature and characteristics of neutron stars",
                "How do credit card rewards influence spending behavior",
                "Investigate the concept of the Great Filter",
            ],
        ),
    ],
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
