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
    TrajectoryExtensionServer,
    TrajectoryExtensionSpec,
)
from beeai_sdk.a2a.types import AgentMessage
from beeai_sdk.server import Server
from beeai_sdk.server.context import RunContext
from beeai_sdk.server.store.platform_context_store import PlatformContextStore
from granite_core.chat.prompts import ChatPrompts
from granite_core.chat_model import ChatModelFactory
from granite_core.config import settings as core_settings
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
    default_input_modes=["text/plain", "text/markdown"],
    default_output_modes=["text/plain"],
    detail=AgentDetail(
        interaction_mode="multi-turn",
        user_greeting="Hi, I'm Granite! How can I help you?",
        framework="BeeAI",
        author=AgentDetailContributor(name="IBM Research"),
        license="Apache 2.0",
    ),
    skills=[
        AgentSkill(
            id="chat",
            name="Chat",
            description="Chat with the model that's enabled with Internet connected search",
            tags=["chat"],
        )
    ],
)
async def chat(
    input: A2AMessage,
    context: RunContext,
    trajectory: Annotated[TrajectoryExtensionServer, TrajectoryExtensionSpec()],
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

        # trajectory message: search start
        metadata = trajectory.trajectory_metadata(title="Searching the web", content="complete")
        yield metadata
        await context.store(AgentMessage(metadata=metadata))

        async with chat_pool.throttle():
            async for event, _ in chat_model.create(messages=messages, stream=True):
                if isinstance(event, ChatModelNewTokenEvent):
                    agent_response_text = event.value.get_text_content()
                    agent_message = AgentMessage(text=agent_response_text)
                    yield agent_message
                    await context.store(agent_message)
                    final_agent_response_text += agent_response_text
        logger.info(f"Agent: {final_agent_response_text}")
    except BaseException as e:
        logger.exception("Chat agent error, threw exception...")
        yield AgentMessage(text=str(e))


if __name__ == "__main__":
    server.run(
        host=settings.HOST,
        port=settings.PORT,
        access_log=settings.ACCESS_LOG,
        context_store=PlatformContextStore(),
    )
