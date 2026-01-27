# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0

import ast
import json
from collections.abc import AsyncGenerator
from logging import Logger
from typing import Annotated, Any

from a2a.types import AgentSkill, Message
from a2a.types import Message as A2AMessage
from a2a.utils.message import get_message_text
from agentstack_sdk.a2a.extensions import (
    CitationExtensionServer,
    CitationExtensionSpec,
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
from langchain_core.tools.base import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from a2a_agents import __version__
from a2a_agents.agents.deep.factory import create_deep_agent
from a2a_agents.agents.deep.middleware import PatchInvalidToolCallsMiddleware
from a2a_agents.agents.deep.prompts import system_prompt
from a2a_agents.agents.deep.util import to_langchain_messages
from a2a_agents.config import agent_detail, settings

RITS_MODEL = ""
RITS_BASE_URL = ""
RITS_API_KEY = ""

logger: Logger = get_logger(logger_name=__name__)
server: Server = Server()


research_skill: AgentSkill = AgentSkill(
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
    user_message: str = get_message_text(message=input)
    logger.info(msg=f"User: {user_message}")

    await context.store(data=input)
    history: list[Message] = [
        message async for message in context.load_history() if isinstance(message, A2AMessage) and message.parts
    ]
    messages: list[AIMessage | HumanMessage] = to_langchain_messages(history)

    model: ChatOpenAI = ChatOpenAI(
        model=RITS_MODEL,
        stream_usage=True,
        temperature=0,
        api_key=SecretStr(secret_value=RITS_API_KEY),
        base_url=RITS_BASE_URL + "/v1",
        default_headers={"RITS_API_KEY": RITS_API_KEY},
    )

    # model = init_chat_model(model="ollama:ibm/granite4", streaming=True)

    mcp_client: MultiServerMCPClient = MultiServerMCPClient(
        connections={
            "internet_search": {
                "url": "http://localhost:8001/mcp",
                "transport": "streamable_http",
            }
        }
    )

    mcp_tools: list[BaseTool] = await mcp_client.get_tools()
    agent = create_deep_agent(
        model=model,
        tools=mcp_tools,
        system_prompt=system_prompt(),
        middleware=[PatchInvalidToolCallsMiddleware()],
        debug=True,
    )

    tool_call: dict[str, str] = {"name": "", "args": ""}

    async for stream_mode, chunk in agent.astream(input={"messages": messages}, stream_mode=["messages"]):
        if stream_mode == "messages":
            msg, _ = tuple(chunk)

            if isinstance(msg, AIMessageChunk):
                if "finish_reason" in msg.response_metadata and msg.response_metadata["finish_reason"] == "tool_calls":
                    # Convert tool args to json
                    args: str = tool_call["args"]
                    if args.startswith('"') and args.endswith('"'):
                        args = tool_call["args"][1:-1].encode().decode(encoding="unicode_escape")
                    try:
                        parsed = json.loads(s=args)
                    except json.JSONDecodeError:
                        parsed = ast.literal_eval(node_or_string=args)

                    tool_call_metadata: Metadata[str, Any] = trajectory.trajectory_metadata(
                        title=tool_call["name"], content=json.dumps(obj=parsed, indent=4)
                    )
                    yield tool_call_metadata
                    await context.store(data=AgentMessage(metadata=tool_call_metadata))
                    tool_call = {"name": "", "args": ""}
                elif msg.tool_call_chunks:
                    # print(event)
                    for tc in msg.tool_call_chunks:
                        tool_call["name"] += tc.get("name") or ""
                        tool_call["args"] += tc.get("args") or ""
                elif msg.text:
                    yield AgentMessage(text=msg.text)
                    await context.store(AgentMessage(text=msg.text))

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
