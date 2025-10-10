from collections.abc import AsyncGenerator

from a2a.types import AgentSkill
from a2a.types import Message as A2AMessage
from a2a.utils.message import get_message_text
from beeai_framework.backend import ChatModelNewTokenEvent, UserMessage
from beeai_sdk.a2a.extensions import (
    AgentDetail,
    AgentDetailContributor,
)
from beeai_sdk.a2a.types import AgentMessage
from beeai_sdk.server import Server
from beeai_sdk.server.context import RunContext
from beeai_sdk.server.store.platform_context_store import PlatformContextStore
from granite_core.chat_model import ChatModelFactory
from granite_core.config import settings as core_settings
from granite_core.logging import get_logger
from granite_core.utils import log_settings
from granite_core.work import chat_pool

from a2a_agents import __version__
from a2a_agents.config import settings
from a2a_agents.utils import to_framework_messages

logger = get_logger(__name__)
log_settings(settings, name="Agent")
log_settings(core_settings)

server = Server()


@server.agent(
    name="Granite Chat",
    description="This agent leverages the IBM Granite models for general chat.",
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
            description="Chat with the model with no external influence",
            tags=["chat"],
        )
    ],
)
async def chat(
    input: A2AMessage,
    context: RunContext,
) -> AsyncGenerator:
    user_message = get_message_text(input)
    logger.info(f"User: {user_message}")

    await context.store(input)
    history = [message async for message in context.load_history() if isinstance(message, A2AMessage) and message.parts]
    messages = to_framework_messages(history)
    messages.append(UserMessage(user_message))

    try:
        final_agent_response_text = ""
        chat_model = ChatModelFactory.create()
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
