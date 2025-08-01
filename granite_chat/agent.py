from collections.abc import AsyncGenerator

from acp_sdk import Annotations, Author, Capability, MessagePart, Metadata
from acp_sdk.models.models import Message, TrajectoryMetadata
from acp_sdk.models.platform import AgentToolInfo, PlatformUIAnnotation, PlatformUIType
from acp_sdk.server import Context, Server
from beeai_framework.backend import (
    ChatModelNewTokenEvent,
    ChatModelSuccessEvent,
    SystemMessage,
)
from beeai_framework.backend import Message as FrameworkMessage
from langchain_core.documents import Document

from granite_chat import get_logger, utils
from granite_chat.chat import ChatModelService
from granite_chat.citations.citations import CitationGeneratorFactory
from granite_chat.config import settings
from granite_chat.emitter import Event
from granite_chat.events import CitationEvent, TextEvent, TrajectoryEvent
from granite_chat.memory import exceeds_token_limit, token_limit_message_part
from granite_chat.research.researcher import Researcher
from granite_chat.search.agent import SearchAgent
from granite_chat.search.embeddings.tokenizer import EmbeddingsTokenizer
from granite_chat.search.prompts import SearchPrompts
from granite_chat.thinking.prompts import ThinkingPrompts
from granite_chat.thinking.response_parser import ThinkingResponseParser
from granite_chat.thinking.stream_handler import TagStartEvent, ThinkingStreamHandler, TokenEvent
from granite_chat.usage import create_usage_info

logger = get_logger(__name__)

# This will preload the embeddings tokenizer if set
EmbeddingsTokenizer.get_instance()

server = Server()

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
    messages = utils.to_beeai_framework_messages(messages=history + input)

    if exceeds_token_limit(messages):
        yield token_limit_message_part()
        return

    chat_model = ChatModelService()

    if settings.STREAMING is True:
        async for event in chat_model.create_stream(messages=messages):
            match event:
                case ChatModelNewTokenEvent():
                    yield MessagePart(content_type="text/plain", content=event.value.get_text_content())
                case (ChatModelSuccessEvent(), "success"):
                    yield create_usage_info(event.value.usage, event.model_id)
    else:
        output = await chat_model.create(messages=messages)
        yield MessagePart(content_type="text/plain", content=output.get_text_content())
        yield create_usage_info(output.usage, chat_model.model_id)


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
    messages = utils.to_beeai_framework_messages(messages=history + input)

    if exceeds_token_limit(messages):
        yield token_limit_message_part()
        return

    chat_model = ChatModelService()

    messages = [
        SystemMessage(content=ThinkingPrompts.granite3_3_thinking_system_prompt()),
        *messages,
    ]

    handler = ThinkingStreamHandler(tags=["think", "response"])

    if settings.STREAMING is True:
        async for event in chat_model.create_stream(messages=messages):
            match event:
                case ChatModelNewTokenEvent():
                    token = event.value.get_text_content()

                    for output in handler.on_token(token=token):
                        if isinstance(output, TokenEvent):
                            content_type = "text/thinking" if output.tag == "think" else "text/plain"
                            yield MessagePart(content_type=content_type, content=output.token)
                        elif isinstance(output, TagStartEvent):
                            pass
                case ChatModelSuccessEvent():
                    yield create_usage_info(event.value.usage, chat_model.model_id)
    else:
        chat_output = await chat_model.create(messages=messages)
        text = chat_output.get_text_content()

        parser = ThinkingResponseParser()
        think_resp = parser.parse(text=text)

        if think_resp.thinking is not None:
            yield MessagePart(content_type="text/thinking", content=think_resp.thinking)

        if think_resp.response is not None:
            yield MessagePart(content_type="text/plain", content=think_resp.response)

        yield create_usage_info(chat_output.usage, chat_model.model_id)


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
        messages = utils.to_beeai_framework_messages(messages=history + input)

        if exceeds_token_limit(messages):
            yield token_limit_message_part()
            return

        chat_model = ChatModelService()
        await context.yield_async(MessagePart(metadata=TrajectoryMetadata(message="Searching the web")))

        search_agent = SearchAgent(chat_model=chat_model)
        docs: list[Document] = await search_agent.search(messages)

        if len(docs) > 0:
            doc_messages: list[FrameworkMessage] = [SystemMessage(content=SearchPrompts.search_system_prompt(docs))]
            # Prepend document prompt
            messages = doc_messages + messages

        response: str = ""

        if settings.STREAMING is True:
            async for event in chat_model.create_stream(messages=messages):
                match event:
                    case ChatModelNewTokenEvent():
                        response += event.value.get_text_content()
                        yield MessagePart(content_type="text/plain", content=event.value.get_text_content())
                    case ChatModelSuccessEvent():
                        yield create_usage_info(event.value.usage, chat_model.model_id)
        else:
            output = await chat_model.create(messages=messages)
            response = output.get_text_content()
            yield MessagePart(content_type="text/plain", content=response)
            yield create_usage_info(output.usage, chat_model.model_id)

        # Yield sources/citation
        if len(docs) > 0:
            generator = CitationGeneratorFactory.create()
            async for citation in generator.generate(messages=input, docs=docs, response=response):
                logger.info(f"Citation: {citation.url}")
                yield utils.to_citation_message_part(citation)

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
        messages = utils.to_beeai_framework_messages(messages=history + input)

        if exceeds_token_limit(messages):
            yield token_limit_message_part()
            return

        chat_model = ChatModelService()

        async def research_listener(event: Event) -> None:
            if isinstance(event, TextEvent):
                await context.yield_async(MessagePart(content=event.text))
            elif isinstance(event, TrajectoryEvent):
                await context.yield_async(MessagePart(metadata=TrajectoryMetadata(message=event.step)))
            elif isinstance(event, CitationEvent):
                logger.info(f"Citation: {event.citation.url}")
                await context.yield_async(utils.to_citation_message_part(event.citation))

        researcher = Researcher(chat_model=chat_model, messages=messages)
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
    async for mp in granite_research(input=input, context=context):
        yield mp


server.run(
    configure_logger=False,
    host=settings.host,
    port=settings.port,
    access_log=settings.ACCESS_LOG,
)
