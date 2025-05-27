import os
import traceback
from collections.abc import AsyncGenerator
from logging import Logger

from acp_sdk import Author, MessagePart, Metadata
from acp_sdk.models import Message
from acp_sdk.server import Context, Server
from beeai_framework.adapters.openai import OpenAIChatModel
from beeai_framework.backend import (
    ChatModelNewTokenEvent,
    ChatModelParameters,
    CustomMessage,
    SystemMessage,
)
from beeai_framework.backend import (
    Message as FrameworkMessage,
)
from config import settings  # type: ignore
from langchain_core.documents import Document

from granite_chat import utils
from granite_chat.search.agent import SearchAgent
from granite_chat.search.prompts import SearchPrompts

logger = Logger("agent", level=settings.log_level)
logger.info(settings)

MODEL_NAME = settings.LLM_MODEL
OPENAI_URL = settings.LLM_API_BASE
OPENAI_API_KEY = settings.LLM_API_KEY

# Allows headers to be picked up by framework
if settings.LLM_API_HEADERS:
    os.environ["OPENAI_API_HEADERS"] = settings.LLM_API_HEADERS

MAX_TOKENS = settings.max_tokens
TEMPERATURE = settings.temperature
SEARCH = settings.search

server = Server()


@server.agent(
    name="Granite Chat",
    description="This agent leverages the Granite 3.3 large language model to deliver fast, accurate, and context-aware conversations. Designed for natural, human-like interaction, the agent can handle complex queries, provide insightful responses, and adapt to a wide range of topics.",  # noqa: E501
    metadata=Metadata(
        ui={"type": "chat", "user_greeting": "Hi, I'm Granite Chat! How can I help you?"},  # type: ignore[call-arg]
        framework="BeeAI",
        programming_language="Python",
        recommended_models=["ibm-granite/granite-3.3-8b-instruct"],
        author=Author(name="IBM Research"),
    ),
)
async def granite_chat(input: list[Message], context: Context) -> AsyncGenerator:

    try:
        # TODO: Manage context window
        messages = utils.to_beeai_framework(messages=input)

        model = OpenAIChatModel(
            model_id=MODEL_NAME,
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_URL,
            parameters=ChatModelParameters(max_tokens=MAX_TOKENS, temperature=TEMPERATURE),
        )

        messages = [
            *messages,
        ]

        if SEARCH:
            search_agent = SearchAgent(chat_model=model)
            docs: list[Document] = await search_agent.search(messages)

            # TODO: Quality control on docs
            # TODO: Better fallback when no good docs found
            if len(docs) > 0:
                doc_messages: list[FrameworkMessage] = [SystemMessage(content=SearchPrompts.search_system_prompt())]

                for i, d in enumerate(docs):
                    role = "document " + str({"document_id": str(i + 1)})
                    logger.info(f"{role} => {d.page_content}")
                    doc_messages.append(CustomMessage(role=role, content=d.page_content))

                # Prepend document prompt and documents
                messages = doc_messages + messages

    except Exception as e:
        traceback.print_exc()
        raise e

    async for data, event in model.create(messages=messages, stream=True):
        match (data, event.name):
            case (ChatModelNewTokenEvent(), "new_token"):
                yield MessagePart(content_type="text/plain", content=data.value.get_text_content(), role="assistant")  # type: ignore[call-arg]


server.run(host=settings.host, port=settings.port, log_level=settings.log_level.lower())
