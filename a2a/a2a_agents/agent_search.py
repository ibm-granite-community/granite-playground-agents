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
    TrajectoryExtensionServer,
    TrajectoryExtensionSpec,
)
from agentstack_sdk.a2a.types import AgentMessage, RunYield
from agentstack_sdk.server import Server
from agentstack_sdk.server.context import RunContext
from agentstack_sdk.server.store.platform_context_store import PlatformContextStore
from beeai_framework.backend import ChatModelNewTokenEvent, SystemMessage, UserMessage
from beeai_framework.backend import Message as FrameworkMessage
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
from a2a_agents.config import agent_detail, settings
from a2a_agents.utils import to_framework_messages

logger = get_logger(__name__)
server = Server()

search_skill = AgentSkill(
    id="search",
    name="Search",
    description="Chat with the model that's enabled with Internet connected search",
    tags=["chat", "search"],
    examples=[
        "What are the latest innovations in long term memory for LLMs?",
        "What is the LLM usage policy at NeurIPS?",
        "What are the best cities to work remotely from for a month — with reliable Wi-Fi, affordable rentals, reliable infrastructure, and good coffee shops or co-working spaces? Focus on destinations in Europe.",  # noqa: E501
    ],
)


@server.agent(
    name="Granite Search",
    description="This agent leverages the IBM Granite models and Internet connected search.",
    version=__version__,
    detail=agent_detail,
    skills=[search_skill],
)
async def agent(
    input: A2AMessage,
    context: RunContext,
    trajectory: Annotated[TrajectoryExtensionServer, TrajectoryExtensionSpec()],
    citation: Annotated[CitationExtensionServer, CitationExtensionSpec()],
) -> AsyncGenerator[RunYield, A2AMessage]:
    # this allows provision of an undecorated search function that can be imported elsewhere
    async for response in search(input, context, trajectory, citation):
        yield response


async def search(
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
        final_agent_response_text: list[str] = []
        final_citations: list[Citation] = []

        # set up chat models
        chat_model = ChatModelFactory.create()
        structured_chat_model = ChatModelFactory.create(model_type="structured")

        # trajectory message: search start
        yield trajectory.trajectory_metadata(title="Searching the web", content="starting")

        # run the search
        search_tool = SearchTool(chat_model=structured_chat_model, session_id=context.context_id)
        docs: list[Document] = await search_tool.search(messages)

        if len(docs) > 0:
            doc_messages: list[FrameworkMessage] = [SystemMessage(content=SearchPrompts.search_system_prompt(docs))]
            # Prepend document prompt
            messages = doc_messages + messages
        else:
            messages = [SystemMessage(content=ChatPrompts.chat_system_prompt()), *messages]

        # trajectory message: search complete
        yield trajectory.trajectory_metadata(title="Searching the web", content="complete")

        # yield response
        async with chat_pool.throttle():
            async for event, _ in chat_model.create(messages=messages, stream=True):
                if isinstance(event, ChatModelNewTokenEvent):
                    agent_response_text = event.value.get_text_content()
                    yield AgentMessage(text=agent_response_text)
                    final_agent_response_text.append(agent_response_text)

        logger.info(f"Agent: {''.join(final_agent_response_text)}")

        # Yield citations
        if len(docs) > 0:
            yield trajectory.trajectory_metadata(title="Generating citations", content="starting")

            # generate citations
            async def citation_handler(event: Event) -> None:
                if isinstance(event, CitationEvent):
                    logger.info(f"Citation: {event.citation.url}")

                    cite = Citation(
                        url=event.citation.url,
                        title=event.citation.title,
                        description=event.citation.context_text,
                        start_index=event.citation.start_index,
                        end_index=event.citation.end_index,
                    )

                    await context.yield_async(citation.citation_metadata(citations=[cite]))
                    final_citations.append(cite)

            generator = CitationGeneratorFactory.create()
            generator.subscribe(handler=citation_handler)
            await generator.generate(docs=docs, response="".join(final_agent_response_text))

            yield trajectory.trajectory_metadata(title="Generating citations", content="complete")

        await context.store(
            AgentMessage(
                text="".join(final_agent_response_text), metadata=citation.citation_metadata(citations=final_citations)
            )
        )

    except BaseException as e:
        logger.exception("Search agent error, threw exception...")
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
