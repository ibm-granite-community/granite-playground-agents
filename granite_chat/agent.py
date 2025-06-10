import logging
import os
import traceback
from collections.abc import AsyncGenerator

from acp_sdk import Author, MessagePart, Metadata
from acp_sdk.models import Message
from acp_sdk.server import Context, Server
from beeai_framework.adapters.openai import OpenAIChatModel
from beeai_framework.backend import (
    ChatModelNewTokenEvent,
    ChatModelParameters,
    SystemMessage,
)
from beeai_framework.backend import Message as FrameworkMessage
from config import settings  # type: ignore
from langchain_core.documents import Document

from granite_chat import utils
from granite_chat.logger import get_formatted_logger
from granite_chat.memory import estimate_tokens
from granite_chat.search.agent import SearchAgent
from granite_chat.search.citations import (
    CitationGenerator,
    DefaultCitationGenerator,
    GraniteIOCitationGenerator,
)
from granite_chat.search.prompts import SearchPrompts
from granite_chat.thinking.prompts import ThinkingPrompts
from granite_chat.thinking.stream_handler import TagStartEvent, ThinkingStreamHandler, TokenEvent
from granite_chat.workers import WorkerPool

logger = get_formatted_logger(__name__, logging.INFO)

MODEL_NAME = settings.LLM_MODEL
OPENAI_URL = settings.LLM_API_BASE
OPENAI_API_KEY = settings.LLM_API_KEY

# Allows headers to be picked up by framework
if settings.LLM_API_HEADERS:
    os.environ["OPENAI_API_HEADERS"] = settings.LLM_API_HEADERS

MAX_TOKENS = settings.MAX_TOKENS
TEMPERATURE = settings.TEMPERATURE
SEARCH = settings.SEARCH
THINKING = settings.THINKING

server = Server()
worker_pool = WorkerPool()


@server.agent(
    name=settings.agent_name,
    description="This agent leverages the Granite 3.3 large language model to deliver fast, accurate, and context-aware conversations. Designed for natural, human-like interaction, the agent can handle complex queries, provide insightful responses, and adapt to a wide range of topics.",  # noqa: E501
    metadata=Metadata(
        ui={"type": "chat", "user_greeting": "Hi, I'm Granite Chat! How can I help you?"},  # type: ignore[call-arg]
        framework="BeeAI",
        programming_language="Python",
        recommended_models=["ibm-granite/granite-3.3-8b-instruct"],
        author=Author(name="IBM Research"),
        env=[
            {"name": "GOOGLE_API_KEY", "description": "Google search API Key"},
            {"name": "GOOGLE_CX_KEY", "description": "Google search engine ID"},
            {"name": "WATSONX_API_BASE", "description": "Watsonx api base url"},
            {"name": "WATSONX_PROJECT_ID", "description": "Watsonx project id"},
            {"name": "WATSONX_REGION", "description": "Watsonx region e.g us-south"},
            {"name": "WATSONX_API_KEY", "description": "Watsonx api key"},
        ],
    ),
)
async def granite_chat(input: list[Message], context: Context) -> AsyncGenerator:
    try:
        # TODO: Manage context window
        messages = utils.to_beeai_framework(messages=input)
        token_count = estimate_tokens(messages)

        if token_count >= settings.CHAT_TOKEN_LIMIT:
            yield MessagePart(
                content_type="limit",
                content="Your message will exceed the length limit for this chat.",
                role="system",
            )  # type: ignore[call-arg]
            return

        model = OpenAIChatModel(
            model_id=MODEL_NAME,
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_URL,
            parameters=ChatModelParameters(max_tokens=MAX_TOKENS, temperature=TEMPERATURE),
        )

        if SEARCH:
            yield MessagePart(content_type="log", content="**Searching the web...**\n\n")

            search_agent = SearchAgent(chat_model=model, worker_pool=worker_pool)
            docs: list[Document] = await search_agent.search(messages)

            if len(docs) > 0:
                doc_messages: list[FrameworkMessage] = [SystemMessage(content=SearchPrompts.search_system_prompt(docs))]
                # Prepend document prompt
                messages = doc_messages + messages

            response: str = ""
            async for data, event in model.create(messages=messages, stream=True):
                match (data, event.name):
                    case (ChatModelNewTokenEvent(), "new_token"):
                        response += data.value.get_text_content()
                        yield MessagePart(
                            content_type="text/plain", content=data.value.get_text_content(), role="assistant"
                        )  # type: ignore[call-arg]

            # Yield sources/citation
            if len(docs) > 0:
                generator: CitationGenerator

                if settings.GRANITE_IO_OPENAI_API_BASE and settings.GRANITE_IO_CITATIONS_MODEL_ID:
                    extra_headers = (
                        dict(
                            pair.split("=", 1) for pair in settings.GRANITE_IO_OPENAI_API_HEADERS.strip('"').split(",")
                        )
                        if settings.GRANITE_IO_OPENAI_API_HEADERS
                        else None
                    )

                    generator = GraniteIOCitationGenerator(
                        openai_base_url=settings.GRANITE_IO_OPENAI_API_BASE,
                        model_id=settings.GRANITE_IO_CITATIONS_MODEL_ID,
                        extra_headers=extra_headers,
                    )
                else:
                    generator = DefaultCitationGenerator()

                async for message_part in generator.generate(messages=input, docs=docs, response=response):
                    yield message_part

        elif THINKING:
            # Prepend a thinking prompt to set the LLM into thinking mode
            messages = [
                SystemMessage(content=ThinkingPrompts.granite3_3_thinking_system_prompt()),
                *messages,
            ]

            handler = ThinkingStreamHandler(tags=["think", "response"])

            async for data, event in model.create(messages=messages, stream=True):
                match (data, event.name):
                    case (ChatModelNewTokenEvent(), "new_token"):
                        token = data.value.get_text_content()
                        for output in handler.on_token(token=token):
                            if isinstance(output, TokenEvent):
                                content_type = "text/thinking" if output.tag == "think" else "text/plain"
                                yield MessagePart(content_type=content_type, content=output.token, role="assistant")  # type: ignore[call-arg]
                            elif isinstance(output, TagStartEvent):
                                if output.tag == "think":
                                    yield MessagePart(
                                        content_type="text/thinking", content="**🤔 Thinking:**\n\n", role="assistant"
                                    )  # type: ignore[call-arg]
                                elif output.tag == "response":
                                    yield MessagePart(
                                        content_type="text/thinking",
                                        content="\n\n**😎 Response:**\n\n",
                                        role="assistant",
                                    )  # type: ignore[call-arg]
        else:
            async for data, event in model.create(messages=messages, stream=True):
                match (data, event.name):
                    case (ChatModelNewTokenEvent(), "new_token"):
                        token = data.value.get_text_content()
                        yield MessagePart(content_type="text/plain", content=token, role="assistant")  # type: ignore[call-arg]

    except Exception as e:
        traceback.print_exc()
        raise e


server.run(host=settings.host, port=settings.port, log_level=settings.log_level.lower())
