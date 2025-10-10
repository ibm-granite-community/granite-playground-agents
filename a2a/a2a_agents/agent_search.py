from collections.abc import AsyncGenerator
from typing import Annotated

from a2a.types import AgentSkill
from a2a.types import Message as A2AMessage
from a2a.utils.message import get_message_text
from beeai_framework.backend import ChatModelNewTokenEvent, SystemMessage, UserMessage
from beeai_framework.backend import Message as FrameworkMessage
from beeai_sdk.a2a.extensions import (
    AgentDetail,
    AgentDetailContributor,
    Citation,
    CitationExtensionServer,
    CitationExtensionSpec,
    TrajectoryExtensionServer,
    TrajectoryExtensionSpec,
)
from beeai_sdk.a2a.types import AgentMessage
from beeai_sdk.server import Server
from beeai_sdk.server.context import RunContext
from beeai_sdk.server.store.platform_context_store import PlatformContextStore
from granite_core.chat.prompts import ChatPrompts
from granite_core.chat_model import ChatModelFactory
from granite_core.citations.citations import CitationGeneratorFactory
from granite_core.citations.events import CitationEvent
from granite_core.config import settings as core_settings
from granite_core.emitter import Event
from granite_core.logging import get_logger
from granite_core.search.prompts import SearchPrompts
from granite_core.search.tool import SearchTool
from granite_core.utils import log_settings
from granite_core.work import chat_pool
from langchain_core.documents import Document

from a2a_agents import __version__
from a2a_agents.config import settings
from a2a_agents.utils import to_framework_messages

logger = get_logger(__name__)
log_settings(settings, name="Agent")
log_settings(core_settings)

server = Server()


@server.agent(
    name="Granite Search",
    description="This agent leverages the IBM Granite models and Internet connected search.",
    documentation_url="https://github.ibm.com/research-design-tech-experiences/granite-agents/",
    version=__version__,
    detail=AgentDetail(
        interaction_mode="multi-turn",
        user_greeting="Hi, I'm Granite! How can I help you?",
        framework="BeeAI",
        author=AgentDetailContributor(name="IBM Research"),
        license="Apache 2.0",
    ),
    skills=[
        AgentSkill(
            id="search",
            name="Search",
            description="Chat with the model that's enabled with Internet connected search",
            tags=["chat", "search"],
        )
    ],
)
async def search(
    input: A2AMessage,
    context: RunContext,
    trajectory: Annotated[TrajectoryExtensionServer, TrajectoryExtensionSpec()],
    citation: Annotated[CitationExtensionServer, CitationExtensionSpec()],
) -> AsyncGenerator:
    user_message = get_message_text(input)
    logger.info(f"User: {user_message}")

    await context.store(input)
    history = [message async for message in context.load_history() if isinstance(message, A2AMessage) and message.parts]
    messages = to_framework_messages(history)
    messages.append(UserMessage(user_message))

    try:
        final_agent_response_text = ""

        # set up chat models
        chat_model = ChatModelFactory.create()
        structured_chat_model = ChatModelFactory.create(model_type="structured")

        # trajectory message: search start
        metadata = trajectory.trajectory_metadata(title="Searching the web", content="starting")
        yield metadata
        await context.store(AgentMessage(metadata=metadata))

        # run the search
        search_tool = SearchTool(chat_model=structured_chat_model, session_id=str(context.context_id))
        docs: list[Document] = await search_tool.search(messages)

        if len(docs) > 0:
            doc_messages: list[FrameworkMessage] = [SystemMessage(content=SearchPrompts.search_system_prompt(docs))]
            # Prepend document prompt
            messages = doc_messages + messages
        else:
            messages = [SystemMessage(content=ChatPrompts.chat_system_prompt()), *messages]

        # trajectory message: search complete
        metadata = trajectory.trajectory_metadata(title="Searching the web", content="complete")
        yield metadata
        await context.store(AgentMessage(metadata=metadata))

        # yield response
        async with chat_pool.throttle():
            async for event, _ in chat_model.create(messages=messages, stream=True):
                if isinstance(event, ChatModelNewTokenEvent):
                    agent_response_text = event.value.get_text_content()
                    agent_message = AgentMessage(text=agent_response_text)
                    yield agent_message
                    await context.store(agent_message)
                    final_agent_response_text += agent_response_text
        logger.info(f"Agent: {final_agent_response_text}")

        # Yield citations
        if len(docs) > 0:
            # trajectory message: citations start
            metadata = trajectory.trajectory_metadata(title="Generating citations", content="starting")
            yield metadata
            await context.store(AgentMessage(metadata=metadata))

            # generate citations
            citations: list[Citation] = []

            async def citation_handler(event: Event) -> None:
                if isinstance(event, CitationEvent):
                    logger.info(f"Citation: {event.citation.url}")
                    citation = Citation(
                        url=event.citation.url,
                        title=event.citation.title,
                        description=event.citation.context_text,
                        start_index=event.citation.start_index,
                        end_index=event.citation.end_index,
                    )
                    citations.append(citation)

            generator = CitationGeneratorFactory.create()
            generator.subscribe(handler=citation_handler)
            await generator.generate(docs=docs, response=final_agent_response_text)

            # yield citations
            message = AgentMessage(
                metadata=(citation.citation_metadata(citations=citations) if citations else None),
            )
            yield message
            await context.store(message)

            # trajectory message: citations complete
            metadata = trajectory.trajectory_metadata(title="Generating citations", content="complete")
            yield metadata
            await context.store(AgentMessage(metadata=metadata))

    except BaseException as e:
        logger.exception("Search agent error, threw exception...")
        yield AgentMessage(text=str(e))


if __name__ == "__main__":
    server.run(
        host=settings.HOST,
        port=settings.PORT,
        access_log=settings.ACCESS_LOG,
        context_store=PlatformContextStore(),
    )
