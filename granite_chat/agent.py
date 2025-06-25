import traceback
from collections.abc import AsyncGenerator

from acp_sdk import Author, MessagePart, Metadata
from acp_sdk.models import Message
from acp_sdk.server import Context, Server
from beeai_framework.backend import (
    ChatModelNewTokenEvent,
    SystemMessage,
)
from beeai_framework.backend import Message as FrameworkMessage
from langchain_core.documents import Document

from granite_chat import utils
from granite_chat.config import settings  # type: ignore
from granite_chat.memory import exceeds_token_limit, token_limit_message_part
from granite_chat.model import ChatModelFactory
from granite_chat.research.researcher import Researcher
from granite_chat.research.types import ResearchEvent
from granite_chat.search.agent import SearchAgent
from granite_chat.search.citations import (
    CitationGenerator,
    DefaultCitationGenerator,
    GraniteIOCitationGenerator,
)
from granite_chat.search.embeddings.tokenizer import EmbeddingsTokenizer
from granite_chat.search.prompts import SearchPrompts
from granite_chat.thinking.prompts import ThinkingPrompts
from granite_chat.thinking.stream_handler import TagStartEvent, ThinkingStreamHandler, TokenEvent
from granite_chat.workers import WorkerPool

LLM_PROVIDER = settings.LLM_PROVIDER

# This will preload the embeddings tokenizer if set
EmbeddingsTokenizer.get_instance()

server = Server()
worker_pool = WorkerPool()


@server.agent(
    name="granite-chat",
    description="This agent leverages the Granite 3.3 large language model for general chat.",
    metadata=Metadata(
        ui={"type": "chat", "user_greeting": "Hi, I'm Granite! How can I help you?"},  # type: ignore[call-arg]
        framework="BeeAI",
        programming_language="Python",
        recommended_models=["ibm-granite/granite-3.3-8b-instruct"],
        author=Author(name="IBM Research"),
        env=[
            {
                "name": "LLM_MODEL",
                "description": "Language model name",
                "required": True,
            },
            {
                "name": "LLM_API_BASE",
                "description": "Base URL of an OpenAI endpoint where the language model is available",
                "required": True,
            },
            {
                "name": "LLM_API_KEY",
                "description": "API Key used to access the OpenAI endpoint",
                "required": True,
            },
        ],
    ),
)
async def granite_chat(input: list[Message], context: Context) -> AsyncGenerator:
    messages = utils.to_beeai_framework(messages=input)

    if exceeds_token_limit(messages):
        yield token_limit_message_part()
        return

    model = ChatModelFactory.create(provider=LLM_PROVIDER)

    async for data, event in model.create(messages=messages, stream=True):
        match (data, event.name):
            case (ChatModelNewTokenEvent(), "new_token"):
                yield MessagePart(content_type="text/plain", content=data.value.get_text_content(), role="assistant")  # type: ignore[call-arg]


@server.agent(
    name="granite-thinking",
    description="This agent leverages the Granite 3.3 large language model for general chat with reasoning.",
    metadata=Metadata(
        ui={"type": "chat", "user_greeting": "Hi, I'm Granite! How can I help you?"},  # type: ignore[call-arg]
        framework="BeeAI",
        programming_language="Python",
        recommended_models=["ibm-granite/granite-3.3-8b-instruct"],
        author=Author(name="IBM Research"),
        env=[
            {
                "name": "LLM_MODEL",
                "description": "Language model name",
                "required": True,
            },
            {
                "name": "LLM_API_BASE",
                "description": "Base URL of an OpenAI endpoint where the language model is available",
                "required": True,
            },
            {
                "name": "LLM_API_KEY",
                "description": "API Key used to access the OpenAI endpoint",
                "required": True,
            },
        ],
    ),
)
async def granite_think(input: list[Message], context: Context) -> AsyncGenerator:
    messages = utils.to_beeai_framework(messages=input)

    if exceeds_token_limit(messages):
        yield token_limit_message_part()
        return

    model = ChatModelFactory.create(provider=LLM_PROVIDER)

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
                                content_type="text/delimiter", content="**🤔 Thinking:**\n\n", role="assistant"
                            )  # type: ignore[call-arg]
                        elif output.tag == "response":
                            yield MessagePart(
                                content_type="text/delimiter",
                                content="\n\n**😎 Response:**\n\n",
                                role="assistant",
                            )  # type: ignore[call-arg]


@server.agent(
    name="granite-search",
    description="This agent leverages the Granite 3.3 large language model to chat and search the web.",
    metadata=Metadata(
        ui={"type": "chat", "user_greeting": "Hi, I'm Granite! How can I help you?"},  # type: ignore[call-arg]
        framework="BeeAI",
        programming_language="Python",
        recommended_models=["ibm-granite/granite-3.3-8b-instruct"],
        author=Author(name="IBM Research"),
        env=[
            {
                "name": "LLM_MODEL",
                "description": "Language model name",
                "required": True,
            },
            {
                "name": "LLM_API_BASE",
                "description": "Base URL of an OpenAI endpoint where the language model is available",
                "required": True,
            },
            {
                "name": "LLM_API_KEY",
                "description": "API Key used to access the OpenAI endpoint",
                "required": True,
            },
            # Only support google for search at the moment
            {
                "name": "GOOGLE_API_KEY",
                "description": "Google search API Key",
            },
            {
                "name": "GOOGLE_CX_KEY",
                "description": "Google search engine ID",
            },
            {
                "name": "TAVILY_API_KEY",
                "description": "Tavily search API key",
            },
            # Embeddings provider
            {"name": "EMBEDDINGS_PROVIDER", "description": "The embeddings provider to use"},
            # For "watsonx" embedding provider
            {"name": "WATSONX_API_BASE", "description": "Watsonx api base url"},
            {"name": "WATSONX_PROJECT_ID", "description": "Watsonx project id"},
            {"name": "WATSONX_REGION", "description": "Watsonx region e.g us-south"},
            {"name": "WATSONX_API_KEY", "description": "Watsonx api key"},
            # For "openai" embedding provider (RITS etc.)
            {"name": "EMBEDDINGS_OPENAI_API_KEY", "description": "OpenAI api key"},
            {"name": "EMBEDDINGS_OPENAI_API_BASE", "description": "OpenAI api base"},
            {"name": "EMBEDDINGS_OPENAI_API_HEADERS", "description": "OpenAI api headers"},
        ],
    ),
)
async def granite_search(input: list[Message], context: Context) -> AsyncGenerator:
    try:
        messages = utils.to_beeai_framework(messages=input)

        if exceeds_token_limit(messages):
            yield token_limit_message_part()
            return

        model = ChatModelFactory.create(provider=LLM_PROVIDER)

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
                    dict(pair.split("=", 1) for pair in settings.GRANITE_IO_OPENAI_API_HEADERS.strip('"').split(","))
                    if settings.GRANITE_IO_OPENAI_API_HEADERS
                    else None
                )

                generator = GraniteIOCitationGenerator(
                    openai_base_url=str(settings.GRANITE_IO_OPENAI_API_BASE),
                    model_id=settings.GRANITE_IO_CITATIONS_MODEL_ID,
                    extra_headers=extra_headers,
                )
            else:
                generator = DefaultCitationGenerator()

            async for message_part in generator.generate(messages=input, docs=docs, response=response):
                yield message_part

    except Exception as e:
        traceback.print_exc()
        raise e


@server.agent(
    name="granite-research",
    description="This agent leverages the Granite 3.3 large language model to perform research.",
    metadata=Metadata(
        ui={"type": "hands-off", "user_greeting": "What topic do you want to research?"},  # type: ignore[call-arg]
        framework="BeeAI",
        programming_language="Python",
        recommended_models=["ibm-granite/granite-3.3-8b-instruct"],
        author=Author(name="IBM Research"),
        env=[
            {
                "name": "LLM_MODEL",
                "description": "Language model name",
                "required": True,
            },
            {
                "name": "LLM_API_BASE",
                "description": "Base URL of an OpenAI endpoint where the language model is available",
                "required": True,
            },
            {
                "name": "LLM_API_KEY",
                "description": "API Key used to access the OpenAI endpoint",
                "required": True,
            },
            # Only support google for search at the moment
            {
                "name": "GOOGLE_API_KEY",
                "description": "Google search API Key",
            },
            {
                "name": "GOOGLE_CX_KEY",
                "description": "Google search engine ID",
            },
            # Embeddings provider
            {"name": "EMBEDDINGS_PROVIDER", "description": "The embeddings provider to use"},
            # For "watsonx" embedding provider
            {"name": "WATSONX_API_BASE", "description": "Watsonx api base url"},
            {"name": "WATSONX_PROJECT_ID", "description": "Watsonx project id"},
            {"name": "WATSONX_REGION", "description": "Watsonx region e.g us-south"},
            {"name": "WATSONX_API_KEY", "description": "Watsonx api key"},
            # For "openai" embedding provider (RITS etc.)
            {"name": "EMBEDDINGS_OPENAI_API_KEY", "description": "OpenAI api key"},
            {"name": "EMBEDDINGS_OPENAI_API_BASE", "description": "OpenAI api base"},
            {"name": "EMBEDDINGS_OPENAI_API_HEADERS", "description": "OpenAI api headers"},
        ],
    ),
)
async def granite_research(input: list[Message], context: Context) -> AsyncGenerator:
    try:
        messages = utils.to_beeai_framework(messages=input)

        if exceeds_token_limit(messages):
            yield token_limit_message_part()
            return

        model = ChatModelFactory.create(provider=LLM_PROVIDER)

        async def research_listener(event: ResearchEvent) -> None:
            if event.event_type == "token":
                await context.yield_async(MessagePart(content=event.data))
            elif event.event_type == "log":
                await context.yield_async({"message": f"{event.data}\n"})

        researcher = Researcher(
            chat_model=model, messages=messages, listener=research_listener, worker_pool=worker_pool
        )

        await researcher.run()

    except Exception as e:
        traceback.print_exc()
        raise e


server.run(host=settings.host, port=settings.port, log_level=settings.log_level.lower(), access_log=settings.ACCESS_LOG)
