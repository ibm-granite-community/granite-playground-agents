# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0


from collections.abc import AsyncGenerator
from typing import Annotated

from a2a.types import AgentSkill
from a2a.types import Message as A2AMessage
from a2a.utils.message import get_message_text
from beeai_framework.backend import UserMessage
from beeai_sdk.a2a.extensions import (
    Citation,
    CitationExtensionServer,
    CitationExtensionSpec,
    TrajectoryExtensionServer,
    TrajectoryExtensionSpec,
)
from beeai_sdk.a2a.types import AgentMessage, RunYield
from beeai_sdk.server import Server
from beeai_sdk.server.context import RunContext
from beeai_sdk.server.store.platform_context_store import PlatformContextStore
from granite_core.chat_model import ChatModelFactory
from granite_core.citations.events import CitationEvent
from granite_core.config import settings as core_settings
from granite_core.emitter import Event
from granite_core.events import (
    GeneratingCitationsCompleteEvent,
    GeneratingCitationsEvent,
    TextEvent,
    TrajectoryEvent,
)
from granite_core.logging import get_logger
from granite_core.research.researcher import Researcher
from granite_core.utils import log_settings

from a2a_agents import __version__
from a2a_agents.config import agent_detail, settings
from a2a_agents.utils import to_framework_messages

logger = get_logger(__name__)
server = Server()

research_skill = AgentSkill(
    id="research",
    name="Research",
    description="Chat with the model that's enabled with Internet connected deep research",
    tags=["research"],
    examples=[
        "Explain the nature and characteristics of neutron stars",
        "How do credit card rewards influence spending behavior",
        "Investigate the concept of the Great Filter",
    ],
)


@server.agent(
    name="Granite Research",
    description="This agent leverages the IBM Granite models and Internet connected deep research.",
    version=__version__,
    detail=agent_detail,
    skills=[research_skill],
)
async def agent(
    input: A2AMessage,
    context: RunContext,
    trajectory: Annotated[TrajectoryExtensionServer, TrajectoryExtensionSpec()],
    citation: Annotated[CitationExtensionServer, CitationExtensionSpec()],
) -> AsyncGenerator[RunYield, A2AMessage]:
    # this allows provision of an undecorated research function that can be imported elsewhere
    async for response in research(input, context, trajectory, citation):
        yield response


async def research(
    input: A2AMessage,
    context: RunContext,
    trajectory: Annotated[TrajectoryExtensionServer, TrajectoryExtensionSpec()],
    citation: Annotated[CitationExtensionServer, CitationExtensionSpec()],
) -> AsyncGenerator[RunYield, A2AMessage]:
    user_message = get_message_text(input)
    logger.info(f"User: {user_message}")

    await context.store(input)
    history = [message async for message in context.load_history() if isinstance(message, A2AMessage) and message.parts]
    messages = to_framework_messages(history)
    messages.append(UserMessage(user_message))

    try:
        # set up chat models
        chat_model = ChatModelFactory.create()
        structured_chat_model = ChatModelFactory.create(model_type="structured")

        # buffer for citations
        citations: list[Citation] = []

        # output researcher events
        async def research_listener(event: Event) -> None:
            if isinstance(event, TextEvent):
                agent_message = AgentMessage(text=event.text)
                await context.yield_async(agent_message)
                await context.store(agent_message)
            elif isinstance(event, TrajectoryEvent):
                if event.content is None:
                    metadata = trajectory.trajectory_metadata(title=event.title)
                    await context.yield_async(metadata)
                    await context.store(AgentMessage(metadata=metadata))
                else:
                    contents = [event.content] if isinstance(event.content, str) else event.content
                    for content in contents:
                        metadata = trajectory.trajectory_metadata(title=event.title, content=content)
                        await context.yield_async(metadata)
                        await context.store(AgentMessage(metadata=metadata))
            elif isinstance(event, GeneratingCitationsEvent):
                metadata = trajectory.trajectory_metadata(title="Generating citations", content="starting")
                await context.yield_async(metadata)
                await context.store(AgentMessage(metadata=metadata))
            elif isinstance(event, CitationEvent):
                logger.info(f"[granite_research:{context.context_id}] Citation: {event.citation.url}")
                citation = Citation(
                    url=event.citation.url,
                    title=event.citation.title,
                    description=event.citation.context_text,
                    start_index=event.citation.start_index,
                    end_index=event.citation.end_index,
                )
                citations.append(citation)
            elif isinstance(event, GeneratingCitationsCompleteEvent):
                metadata = trajectory.trajectory_metadata(title="Generating citations", content="complete")
                await context.yield_async(metadata)
                await context.store(AgentMessage(metadata=metadata))

        # create and run the researcher
        researcher = Researcher(
            chat_model=chat_model,
            structured_chat_model=structured_chat_model,
            messages=messages,
            session_id=context.context_id,
        )
        researcher.subscribe(handler=research_listener)
        await researcher.run()

        # yield citations
        message = AgentMessage(
            metadata=(citation.citation_metadata(citations=citations) if citations else None),
        )
        yield message
        await context.store(message)

    except BaseException as e:
        logger.exception("Research agent error, threw exception...")
        yield AgentMessage(text=str(e))


if __name__ == "__main__":
    log_settings(settings, name="Agent")
    log_settings(core_settings)
    server.run(
        host=settings.HOST,
        port=settings.PORT,
        access_log=settings.ACCESS_LOG,
        context_store=PlatformContextStore(),
    )
