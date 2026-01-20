# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0


import itertools
import json
from collections import defaultdict
from collections.abc import AsyncGenerator
from logging import Logger
from typing import Annotated, Any

from a2a.types import AgentSkill, Message, Role
from a2a.types import Message as A2AMessage
from a2a.utils.message import get_message_text
from agentstack_sdk.a2a.extensions import (
    CitationExtensionServer,
    CitationExtensionSpec,
    EmbeddingServiceExtensionServer,
    EmbeddingServiceExtensionSpec,
    LLMServiceExtensionServer,
    LLMServiceExtensionSpec,
    TrajectoryExtensionServer,
    TrajectoryExtensionSpec,
)
from agentstack_sdk.a2a.types import AgentMessage, Metadata, RunYield
from agentstack_sdk.server import Server
from agentstack_sdk.server.context import RunContext
from agentstack_sdk.server.store.platform_context_store import PlatformContextStore
from granite_core.config import settings as core_settings
from granite_core.logging import get_logger
from granite_core.utils import log_settings
from langchain.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from a2a_agents import __version__
from a2a_agents.agents.deep_agent.factory import create_deep_agent
from a2a_agents.agents.deep_agent.prompts import system_prompt

# from deepagents import create_deep_agent
from a2a_agents.agents.deep_agent.tools import internet_search
from a2a_agents.config import agent_detail, settings
from a2a_agents.utils import configure_models

logger: Logger = get_logger(logger_name=__name__)
server: Server = Server()

ROLE_TO_MESSAGE: dict[Role, type[HumanMessage] | type[AIMessage]] = {
    Role.user: HumanMessage,
    Role.agent: AIMessage,
}


def to_langchain_messages(history: list[A2AMessage]) -> list[AIMessage | HumanMessage]:
    """
    Converts a list of messages into a list of framework messages, separating user and agent turns.

    Args:
        history (list[Message]): A list of messages containing user and agent turns.

    Returns:
        list[AnyMessage]: A list of framework messages, where each message represents
            either a user or an agent's input, depending on the turn.
    """
    if not history:
        return []

    langchain_messages: list[AIMessage | HumanMessage] = []

    for role, group in itertools.groupby(history, key=lambda msg: msg.role):
        # Collect all text parts from consecutive messages with the same role
        text_parts: list[str] = []
        for message in group:
            text_parts.extend(part.root.text for part in message.parts if part.root.kind == "text")

        # Join all text parts and create the appropriate message type
        combined_text: str = "".join(text_parts)

        if role not in ROLE_TO_MESSAGE:
            raise ValueError(f"Unknown role in message history: {role}")

        message_class: type[HumanMessage] | type[AIMessage] = ROLE_TO_MESSAGE[role]
        langchain_messages.append(message_class(content=combined_text))

    return langchain_messages


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
        name="Granite",
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
            EmbeddingServiceExtensionSpec.single_demand(suggested=(settings.SUGGETED_EMBEDDING_MODEL,)),
        ],
        trajectory: Annotated[TrajectoryExtensionServer, TrajectoryExtensionSpec()],
        citation: Annotated[CitationExtensionServer, CitationExtensionSpec()],
    ) -> AsyncGenerator[RunYield, A2AMessage]:
        # this allows provision of an undecorated research function that can be imported elsewhere
        async for response in _agent(input, context, trajectory, citation, llm_ext, embedding_ext):
            yield response

else:
    # agent without LLM extensions
    @server.agent(
        name="Granite",
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
        async for response in _agent(input, context, trajectory, citation):
            yield response


async def _agent(
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
            EmbeddingServiceExtensionSpec.single_demand(suggested=(settings.SUGGETED_EMBEDDING_MODEL,)),
        ]
        | None
    ) = None,
) -> AsyncGenerator[RunYield, A2AMessage]:
    await configure_models(llm_ext, embedding_ext)
    user_message: str = get_message_text(message=input)
    logger.info(msg=f"User: {user_message}")

    await context.store(data=input)
    history: list[Message] = [
        message async for message in context.load_history() if isinstance(message, A2AMessage) and message.parts
    ]
    messages: list[AIMessage | HumanMessage] = to_langchain_messages(history)
    messages.append(HumanMessage(content=user_message))

    model: ChatOpenAI = ChatOpenAI(
        model="",
        stream_usage=True,
        temperature=0,
        # max_tokens=None,
        # timeout=None,
        # reasoning_effort="low",
        # max_retries=2,
        api_key=SecretStr(secret_value=""),  # If you prefer to pass api key in directly
        base_url="",
        default_headers={"": ""},
        # organization="...",
        # other params...
    )
    # model = init_chat_model(model="ollama:ibm/granite4", streaming=True)
    agent = create_deep_agent(model=model, tools=[internet_search], system_prompt=system_prompt, middleware=[])

    tool_calls: defaultdict[str, dict[str, str]] = defaultdict(lambda: {"name": "", "args": ""})

    for event in agent.stream(input={"messages": messages}, stream_mode=["messages"]):
        # event is a tuple: (node_name, messages_list)
        print(event)
        node_name, messages_list = event
        if node_name == "messages":
            msg: str | Any = messages_list[0]

            if isinstance(msg, AIMessageChunk):
                if "finish_reason" in msg.response_metadata and msg.response_metadata["finish_reason"] == "tool_calls":
                    for _, data in tool_calls.items():
                        tool_call_metadata: Metadata[str, Any] = trajectory.trajectory_metadata(
                            title=data["name"], content=json.dumps(obj=data["args"])
                        )
                        yield tool_call_metadata
                        await context.store(data=AgentMessage(metadata=tool_call_metadata))
                    tool_calls.clear()

                elif msg.tool_call_chunks:
                    for tc in msg.tool_call_chunks:
                        tc_id: str | None = tc.get("id")
                        if tc_id:
                            tool_calls[tc_id]["name"] += tc.get("name") or ""
                            tool_calls[tc_id]["args"] += tc.get("args") or ""
                elif msg.text:
                    yield AgentMessage(text=msg.text)
                    await context.store(AgentMessage(text=msg.text))

            # Tool call response
            elif isinstance(msg, ToolMessage):
                tool_message_metadata: Metadata[str, Any] = trajectory.trajectory_metadata(
                    title=msg.name, content=msg.text
                )
                yield tool_message_metadata
                await context.store(data=AgentMessage(metadata=tool_message_metadata))


if __name__ == "__main__":
    log_settings(settings, name="Agent")
    log_settings(settings=core_settings)
    server.run(
        host=settings.HOST,
        port=settings.PORT,
        access_log=settings.ACCESS_LOG,
        context_store=PlatformContextStore(),
    )
