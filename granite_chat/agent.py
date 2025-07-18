import re
from collections.abc import AsyncGenerator
from typing import Literal, Optional

from acp_sdk import Annotations, Author, Capability, MessagePart, Metadata
from acp_sdk.models import Message
from acp_sdk.models.platform import AgentToolInfo, PlatformUIAnnotation, PlatformUIType
from acp_sdk.server import Context, Server
from beeai_framework.backend import (
    ChatModelNewTokenEvent,
    SystemMessage,
    ChatModelSuccessEvent,
)
from beeai_framework.backend import Message as FrameworkMessage
from beeai_framework.backend.types import ChatModelUsage
from langchain_core.documents import Document
from pydantic import BaseModel

from granite_chat import get_logger, utils
from granite_chat.config import settings
from granite_chat.emitter import Event
from granite_chat.memory import exceeds_token_limit, token_limit_message_part
from granite_chat.model import ChatModelFactory
from granite_chat.research.researcher import Researcher
from granite_chat.search.agent import SearchAgent
from granite_chat.search.citations import CitationGenerator
from granite_chat.search.embeddings.tokenizer import EmbeddingsTokenizer
from granite_chat.search.prompts import SearchPrompts
from granite_chat.thinking.prompts import ThinkingPrompts
from granite_chat.thinking.stream_handler import TagStartEvent, ThinkingStreamHandler, TokenEvent
from granite_chat.workers import WorkerPool

logger = get_logger(__name__)

LLM_PROVIDER = settings.LLM_PROVIDER

# This will preload the embeddings tokenizer if set
EmbeddingsTokenizer.get_instance()

server = Server()
worker_pool = WorkerPool()


base_env = [
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
]

search_env = [
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
]

watsonx_env = [
    {"name": "WATSONX_API_BASE", "description": "Watsonx api base url"},
    {"name": "WATSONX_PROJECT_ID", "description": "Watsonx project id"},
    {"name": "WATSONX_REGION", "description": "Watsonx region e.g us-south"},
    {"name": "WATSONX_API_KEY", "description": "Watsonx api key"},
]

class UsageInfo(BaseModel):
    completion_tokens: Optional[int]
    prompt_tokens: Optional[int]
    total_tokens: Optional[int]
    model_id: str
    type: Literal["usage_info"] = "usage_info"

def create_usage_info(
        usage: Optional[ChatModelUsage],
        model_id: str,
):
    return UsageInfo(
        completion_tokens=usage.completion_tokens if usage else None,
        prompt_tokens=usage.prompt_tokens if usage else None,
        total_tokens=usage.total_tokens if usage else None,
        model_id=model_id,
    )

@server.agent(
    name="granite-chat",
    description="This agent leverages the Granite 3.3 large language model for general chat.",
    metadata=Metadata(
        annotations=Annotations(
            beeai_ui=PlatformUIAnnotation(
                ui_type=PlatformUIType.CHAT, user_greeting="Hi, I'm Granite! How can I help you?", display_name="Chat"
            )
        ),
        programming_language="Python",
        natural_languages=["English"],
        framework="BeeAI",
        capabilities=[Capability(name="Chat", description="Chat with the model with no external influence")],
        author=Author(name="IBM Research"),
        recommended_models=["ibm-granite/granite-3.3-8b-instruct"],
        env=base_env,
    ),  # type: ignore[call-arg]
)
async def granite_chat(input: list[Message], context: Context) -> AsyncGenerator:
    history = [message async for message in context.session.load_history()]
    messages = utils.to_beeai_framework(messages=history + input)

    if exceeds_token_limit(messages):
        yield token_limit_message_part()
        return

    model = ChatModelFactory.create(provider=LLM_PROVIDER)

    if settings.STREAMING is True:
        async for data, event in model.create(messages=messages, stream=True):
            match (data, event.name):
                case (ChatModelNewTokenEvent(), "new_token"):
                    yield MessagePart(
                        content_type="text/plain", content=data.value.get_text_content(), role="assistant"
                    )  # type: ignore[call-arg]
                case (ChatModelSuccessEvent(), "success"):
                    yield create_usage_info(
                        data.value.usage,
                        model.model_id
                    )
    else:
        output = await model.create(messages=messages)
        yield MessagePart(content_type="text/plain", content=output.get_text_content())

        yield create_usage_info(
            output.value.usage,
            model.model_id
        )


@server.agent(
    name="granite-thinking",
    description="This agent leverages the Granite 3.3 large language model for general chat with reasoning.",
    metadata=Metadata(
        annotations=Annotations(
            beeai_ui=PlatformUIAnnotation(
                ui_type=PlatformUIType.CHAT,
                user_greeting="Hi, I'm Granite! How can I help you?",
                display_name="Thinking",
            )
        ),
        programming_language="Python",
        natural_languages=["English"],
        framework="BeeAI",
        capabilities=[
            Capability(
                name="Reasoning",
                description="Usage and explanation of a thinking/reasoning process to self-reflect on generated output",
            ),
        ],
        author=Author(name="IBM Research"),
        recommended_models=["ibm-granite/granite-3.3-8b-instruct"],
        env=base_env,
    ),  # type: ignore[call-arg]
)
async def granite_think(input: list[Message], context: Context) -> AsyncGenerator:
    history = [message async for message in context.session.load_history()]
    messages = utils.to_beeai_framework(messages=history + input)

    if exceeds_token_limit(messages):
        yield token_limit_message_part()
        return

    model = ChatModelFactory.create(provider=LLM_PROVIDER)

    messages = [
        SystemMessage(content=ThinkingPrompts.granite3_3_thinking_system_prompt()),
        *messages,
    ]

    handler = ThinkingStreamHandler(tags=["think", "response"])

    if settings.STREAMING is True:
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
                case (ChatModelSuccessEvent(), "success"):
                    yield create_usage_info(
                        data.value.usage,
                        model.model_id
                    )
    else:
        chat_output = await model.create(messages=messages)
        text = chat_output.get_text_content()

        think_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
        response_match = re.search(r"<response>(.*?)</response>", text, re.DOTALL)

        think_content = think_match.group(1) if think_match else None
        response_content = response_match.group(1) if response_match else None

        if think_content is not None:
            yield MessagePart(content_type="text/delimiter", content="**🤔 Thinking:**\n\n", role="assistant")  # type: ignore[call-arg]
            yield MessagePart(content_type="text/thinking", content=think_content)

        if response_content is not None:
            yield MessagePart(
                content_type="text/delimiter",
                content="\n\n**😎 Response:**\n\n",
                role="assistant",
            )  # type: ignore[call-arg]

            yield MessagePart(content_type="text/plain", content=response_content)

        yield create_usage_info(
            chat_output.value.usage,
            model.model_id
        )


@server.agent(
    name="granite-search",
    description="This agent leverages the Granite 3.3 large language model to chat and search the web.",
    metadata=Metadata(
        annotations=Annotations(
            beeai_ui=PlatformUIAnnotation(
                ui_type=PlatformUIType.CHAT,
                user_greeting="Hi, I'm Granite! How can I help you?",
                display_name="Search",
                tools=[AgentToolInfo(name="Search", description="Search engine")],
            )
        ),
        programming_language="Python",
        natural_languages=["English"],
        framework="BeeAI",
        capabilities=[
            Capability(
                name="Search",
                description="Connects the model to a search engine.",
            ),
        ],
        author=Author(name="IBM Research"),
        recommended_models=["ibm-granite/granite-3.3-8b-instruct"],
        env=[
            *base_env,
            *search_env,
            {"name": "EMBEDDINGS_PROVIDER", "description": "The embeddings provider to use"},
            *watsonx_env,
            {"name": "EMBEDDINGS_OPENAI_API_KEY", "description": "OpenAI api key"},
            {"name": "EMBEDDINGS_OPENAI_API_BASE", "description": "OpenAI api base"},
            {"name": "EMBEDDINGS_OPENAI_API_HEADERS", "description": "OpenAI api headers"},
        ],
    ),  # type: ignore[call-arg]
)
async def granite_search(input: list[Message], context: Context) -> AsyncGenerator:
    try:
        history = [message async for message in context.session.load_history()]
        messages = utils.to_beeai_framework(messages=history + input)

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

        if settings.STREAMING is True:
            async for data, event in model.create(messages=messages, stream=True):
                match (data, event.name):
                    case (ChatModelNewTokenEvent(), "new_token"):
                        response += data.value.get_text_content()
                        yield MessagePart(
                            content_type="text/plain", content=data.value.get_text_content(), role="assistant"
                        )  # type: ignore[call-arg]

                    case (ChatModelSuccessEvent(), "success"):
                        yield create_usage_info(
                            data.value.usage,
                            model.model_id
                        )
        else:
            output = await model.create(messages=messages)
            yield MessagePart(content_type="text/plain", content=output.get_text_content(), role="assistant")  # type: ignore[call-arg]
            yield create_usage_info(
                output.value.usage,
                model.model_id
            )

        # Yield sources/citation
        if len(docs) > 0:
            generator = CitationGenerator.create()

            async for message_part in generator.generate(messages=input, docs=docs, response=response):
                yield message_part

    except Exception as e:
        logger.exception(repr(e))
        raise e


@server.agent(
    name="granite-research",
    description="This agent leverages the Granite 3.3 large language model to perform research.",
    metadata=Metadata(
        annotations=Annotations(
            beeai_ui=PlatformUIAnnotation(
                ui_type=PlatformUIType.CHAT,
                user_greeting="What topic do you want to research?",
                display_name="Research",
                tools=[AgentToolInfo(name="Search", description="Search engine")],
            )
        ),
        programming_language="Python",
        natural_languages=["English"],
        framework="BeeAI",
        capabilities=[
            Capability(
                name="Deep Research",
                description="Connects the model to a search engine to perform deep research.",
            ),
        ],
        author=Author(name="IBM Research"),
        recommended_models=["ibm-granite/granite-3.3-8b-instruct"],
        env=[
            *base_env,
            *search_env,
            {"name": "EMBEDDINGS_PROVIDER", "description": "The embeddings provider to use"},
            *watsonx_env,
            {"name": "EMBEDDINGS_OPENAI_API_KEY", "description": "OpenAI api key"},
            {"name": "EMBEDDINGS_OPENAI_API_BASE", "description": "OpenAI api base"},
            {"name": "EMBEDDINGS_OPENAI_API_HEADERS", "description": "OpenAI api headers"},
        ],
    ),  # type: ignore[call-arg]
)
async def granite_research(input: list[Message], context: Context) -> AsyncGenerator:
    try:
        history = [message async for message in context.session.load_history()]
        messages = utils.to_beeai_framework(messages=history + input)

        if exceeds_token_limit(messages):
            yield token_limit_message_part()
            return

        model = ChatModelFactory.create(provider=LLM_PROVIDER)

        async def research_listener(event: Event) -> None:
            if event.type == "token":
                await context.yield_async(MessagePart(content=event.data))
            elif event.type == "log":
                await context.yield_async(MessagePart(content=f"{event.data}\n\n", content_type="text/log"))

        researcher = Researcher(chat_model=model, messages=messages, worker_pool=worker_pool)
        researcher.subscribe(handler=research_listener)
        await researcher.run()

    except Exception as e:
        logger.exception(repr(e))
        raise e


@server.agent(
    name="granite-research-hands-off",
    description="This agent leverages the Granite 3.3 large language model to perform research.",
    metadata=Metadata(
        annotations=Annotations(
            beeai_ui=PlatformUIAnnotation(
                ui_type=PlatformUIType.HANDSOFF,
                user_greeting="What topic do you want to research?",
                display_name="Granite Researcher",
                tools=[AgentToolInfo(name="Search", description="Search engine")],
            )
        ),
        programming_language="Python",
        natural_languages=["English"],
        framework="BeeAI",
        capabilities=[
            Capability(
                name="Deep Research",
                description="Connects the model to a search engine to perform deep research.",
            ),
        ],
        author=Author(name="IBM Research"),
        recommended_models=["ibm-granite/granite-3.3-8b-instruct"],
        env=[
            *base_env,
            *search_env,
            {"name": "EMBEDDINGS_PROVIDER", "description": "The embeddings provider to use"},
            *watsonx_env,
            {"name": "EMBEDDINGS_OPENAI_API_KEY", "description": "OpenAI api key"},
            {"name": "EMBEDDINGS_OPENAI_API_BASE", "description": "OpenAI api base"},
            {"name": "EMBEDDINGS_OPENAI_API_HEADERS", "description": "OpenAI api headers"},
        ],
    ),  # type: ignore[call-arg]
)
async def granite_research_hands_off(input: list[Message], context: Context) -> AsyncGenerator:
    try:
        messages = utils.to_beeai_framework(messages=input)

        if exceeds_token_limit(messages):
            yield token_limit_message_part()
            return

        model = ChatModelFactory.create(provider=LLM_PROVIDER)

        async def research_listener(event: Event) -> None:
            if event.type == "token":
                await context.yield_async(MessagePart(content=event.data))
            elif event.type == "log":
                await context.yield_async({"message": f"{event.data}\n"})

        researcher = Researcher(chat_model=model, messages=messages, worker_pool=worker_pool)
        researcher.subscribe(handler=research_listener)

        # Research report will be yielded
        await researcher.run()

    except Exception as e:
        logger.exception(repr(e))
        raise e


server.run(
    configure_logger=False,
    host=settings.host,
    port=settings.port,
    access_log=settings.ACCESS_LOG,
)
