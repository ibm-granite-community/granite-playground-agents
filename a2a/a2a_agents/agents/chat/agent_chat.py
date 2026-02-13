# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0


from collections.abc import AsyncGenerator
from typing import Annotated

from a2a.types import AgentSkill, Message
from a2a.types import Message as A2AMessage
from a2a.utils.message import get_message_text
from agentstack_sdk.a2a.extensions import (
    LLMServiceExtensionServer,
    LLMServiceExtensionSpec,
)
from agentstack_sdk.a2a.types import AgentMessage, RunYield
from agentstack_sdk.server import Server
from agentstack_sdk.server.context import RunContext
from agentstack_sdk.server.store.platform_context_store import PlatformContextStore
from beeai_framework.backend import UserMessage
from beeai_framework.backend.chat import ChatModel
from beeai_framework.backend.message import AnyMessage
from granite_core.chat.handler import ChatHandler
from granite_core.chat_model import ChatModelFactory
from granite_core.config import settings as core_settings
from granite_core.emitter import Event
from granite_core.events import TextEvent
from granite_core.logging import get_logger
from granite_core.utils import log_settings

from a2a_agents import __version__
from a2a_agents.config import agent_detail, settings
from a2a_agents.utils import configure_models, to_framework_messages

logger = get_logger(__name__)
server = Server()

chat_skill = AgentSkill(
    id="chat",
    name="Chat",
    description="Chat with the model with no external influence",
    tags=["chat"],
    examples=[
        "Explain how RSUs work. I just got offered RSUs in a job offer — what exactly are they and how do they pay out? Break down how RSUs work, including how they vest, the tax implications at each stage, and how I might think about their long-term value compared to salary or stock options.",  # noqa: E501
        "Write a thank you note to my colleague Elaine who is celebrating her 10-year work anniversary. Make it short but mention her important contributions to the Granite project and her valuable mentorship of new team members.",  # noqa: E501
        "Help me ideate compelling and distinctive project names for my research work on autonomous web agents. The names should feel futuristic but grounded, relevant to AI or web infrastructure, and easy to remember. Provide 5 options with short rationale for each.",  # noqa: E501
    ],
)

if settings.USE_AGENTSTACK_LLM:
    # agent with LLM extensions
    @server.agent(
        name="Granite Chat",
        description="This agent leverages the IBM Granite models for general chat.",
        version=__version__,
        detail=agent_detail,
        skills=[chat_skill],
    )
    async def agent(
        input: A2AMessage,
        context: RunContext,
        llm_ext: Annotated[
            LLMServiceExtensionServer,
            LLMServiceExtensionSpec.single_demand(suggested=(settings.SUGGESTED_LLM_MODEL,)),
        ],
    ) -> AsyncGenerator[RunYield, A2AMessage]:
        # this allows provision of an undecorated chat function that can be imported elsewhere
        async for response in chat(input, context, llm_ext):
            yield response

else:
    # agent without LLM extensions
    @server.agent(
        name="Granite Chat",
        description="This agent leverages the IBM Granite models for general chat.",
        version=__version__,
        detail=agent_detail,
        skills=[chat_skill],
    )
    async def agent(
        input: A2AMessage,
        context: RunContext,
    ) -> AsyncGenerator[RunYield, A2AMessage]:
        # this allows provision of an undecorated chat function that can be imported elsewhere
        async for response in chat(input, context):
            yield response


async def chat(
    input: A2AMessage,
    context: RunContext,
    llm_ext: (
        Annotated[
            LLMServiceExtensionServer,
            LLMServiceExtensionSpec.single_demand(suggested=(settings.SUGGESTED_LLM_MODEL,)),
        ]
        | None
    ) = None,
) -> AsyncGenerator[RunYield, A2AMessage]:
    await configure_models(llm_ext)

    user_message: str = get_message_text(message=input)
    logger.info(msg=f"User: {user_message}")

    await context.store(data=input)
    history: list[Message] = [
        message async for message in context.load_history() if isinstance(message, A2AMessage) and message.parts
    ]
    messages: list[AnyMessage] = to_framework_messages(history)
    messages.append(UserMessage(user_message))

    try:
        chat_model: ChatModel = ChatModelFactory.create()
        chat_handler: ChatHandler = ChatHandler(chat_model=chat_model, session_id=context.context_id)
        agent_response_text: list[str] = []

        # output chat events
        async def chat_listener(event: Event) -> None:
            if isinstance(event, TextEvent):
                await context.yield_async(value=AgentMessage(text=event.text))
                agent_response_text.append(event.text)

        chat_handler.subscribe(handler=chat_listener)
        await chat_handler.run(messages, stream=core_settings.STREAMING)

        logger.info(msg=f"Agent: {''.join(agent_response_text)}")
        await context.store(data=AgentMessage(text="".join(agent_response_text)))

    except BaseException as e:
        logger.exception(msg="Chat agent error, threw exception...")
        error_msg: str = f"Error processing request: {e!s}"
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
