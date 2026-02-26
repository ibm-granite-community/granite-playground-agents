# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0


from collections.abc import AsyncGenerator
from typing import Annotated

from a2a.types import AgentSkill
from a2a.types import Message as A2AMessage
from a2a.utils.message import get_message_text
from agentstack_sdk.a2a.extensions import (
    Citation,
    CitationExtensionServer,
    CitationExtensionSpec,
    EmbeddingServiceExtensionServer,
    EmbeddingServiceExtensionSpec,
    LLMServiceExtensionServer,
    LLMServiceExtensionSpec,
    TrajectoryExtensionServer,
    TrajectoryExtensionSpec,
)
from agentstack_sdk.a2a.types import AgentMessage, RunYield
from agentstack_sdk.server import Server
from agentstack_sdk.server.context import RunContext
from agentstack_sdk.server.store.platform_context_store import PlatformContextStore
from beeai_framework.backend import UserMessage
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
from a2a_agents.trajectory import TrajectoryHandler
from a2a_agents.utils import configure_models, to_framework_messages

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


if settings.USE_AGENTSTACK_LLM:
    # agent with LLM extensions
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
        llm_ext: Annotated[
            LLMServiceExtensionServer,
            LLMServiceExtensionSpec.single_demand(suggested=(settings.SUGGESTED_LLM_MODEL,)),
        ],
        embedding_ext: Annotated[
            EmbeddingServiceExtensionServer,
            EmbeddingServiceExtensionSpec.single_demand(suggested=(settings.SUGGESTED_EMBEDDING_MODEL,)),
        ],
        trajectory: Annotated[TrajectoryExtensionServer, TrajectoryExtensionSpec()],
        citation: Annotated[CitationExtensionServer, CitationExtensionSpec()],
    ) -> AsyncGenerator[RunYield, A2AMessage]:
        # this allows provision of an undecorated research function that can be imported elsewhere
        async for response in research(input, context, trajectory, citation, llm_ext, embedding_ext):
            yield response

else:
    # agent without LLM extensions
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
    llm_ext: (
        Annotated[
            LLMServiceExtensionServer,
            LLMServiceExtensionSpec.single_demand(suggested=(settings.SUGGESTED_LLM_MODEL,)),
        ]
        | None
    ) = None,
    embedding_ext: (
        Annotated[
            EmbeddingServiceExtensionServer,
            EmbeddingServiceExtensionSpec.single_demand(suggested=(settings.SUGGESTED_EMBEDDING_MODEL,)),
        ]
        | None
    ) = None,
) -> AsyncGenerator[RunYield, A2AMessage]:
    await configure_models(llm_ext, embedding_ext)

    user_message = get_message_text(input)
    logger.info(f"User: {user_message}")

    await context.store(input)
    history = [message async for message in context.load_history() if isinstance(message, A2AMessage) and message.parts]
    messages = to_framework_messages(history)
    messages.append(UserMessage(user_message))

    trajectory_handler = TrajectoryHandler(trajectory=trajectory, context=context)

    try:
        # set up chat models
        chat_model = ChatModelFactory.create()
        structured_chat_model = ChatModelFactory.create(model_type="structured")
        final_agent_response_text: list[str] = []
        final_citations: list[Citation] = []

        # output researcher events
        async def research_listener(event: Event) -> None:
            if isinstance(event, TextEvent):
                await context.yield_async(AgentMessage(text=event.text))
                final_agent_response_text.append(event.text)
            elif isinstance(event, TrajectoryEvent):
                if event.content is None:
                    await trajectory_handler.yield_trajectory(title=event.title)
                elif isinstance(event.content, str):
                    await trajectory_handler.yield_trajectory(
                        title=event.title, content=event.content, group_id=event.title
                    )
                elif isinstance(event.content, list):
                    await trajectory_handler.yield_trajectory(
                        title=event.title, content="\n".join(f"* {c}" for c in event.content)
                    )
                else:
                    logger.warning(f"Unknown trajectory content type: {type(event.content)}")

            elif isinstance(event, GeneratingCitationsEvent):
                await trajectory_handler.yield_trajectory(
                    title="Generating citations", content="* Starting", group_id="citations"
                )
            elif isinstance(event, CitationEvent):
                logger.info(f"[granite_research:{context.context_id}] Citation: {event.citation.url}")

                cite = Citation(
                    url=event.citation.url,
                    title=event.citation.title,
                    description=event.citation.context_text,
                    start_index=event.citation.start_index,
                    end_index=event.citation.end_index,
                )

                await context.yield_async(citation.citation_metadata(citations=[cite]))
                final_citations.append(cite)

            elif isinstance(event, GeneratingCitationsCompleteEvent):
                await trajectory_handler.yield_trajectory(
                    title="Generating citations", content="Complete", group_id="citations"
                )

        # create and run the researcher
        researcher = Researcher(
            chat_model=chat_model,
            structured_chat_model=structured_chat_model,
            messages=messages,
            session_id=context.context_id,
            interactive=True,
        )
        researcher.subscribe(handler=research_listener)
        await researcher.run()
        await trajectory_handler.store()
        await context.store(
            AgentMessage(
                text="".join(final_agent_response_text), metadata=citation.citation_metadata(citations=final_citations)
            )
        )

    except BaseException as e:
        logger.exception("Research agent error, threw exception...")
        error_msg = f"Error processing request: {e!s}"
        yield error_msg
        await context.store(AgentMessage(text=error_msg))


if __name__ == "__main__":
    log_settings(settings, name="Agent")
    log_settings(core_settings)
    server.run(
        host=settings.HOST,
        port=settings.PORT,
        access_log=settings.ACCESS_LOG,
        context_store=PlatformContextStore(),
    )
