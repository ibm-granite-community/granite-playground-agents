# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0


from collections.abc import AsyncGenerator
from typing import cast

from acp_sdk import Annotations, Author, Capability, MessagePart, Metadata
from acp_sdk.models.models import Message, TrajectoryMetadata
from acp_sdk.models.platform import AgentToolInfo, PlatformUIAnnotation, PlatformUIType
from acp_sdk.server import Context, RedisStore, Server
from beeai_framework.backend import (
    ChatModelNewTokenEvent,
    ChatModelSuccessEvent,
    SystemMessage,
)
from beeai_framework.backend import Message as FrameworkMessage
from granite_core.chat.prompts import ChatPrompts
from granite_core.chat_model import ChatModelFactory
from granite_core.citations.citations import CitationGeneratorFactory
from granite_core.citations.events import CitationEvent
from granite_core.config import settings as core_settings
from granite_core.emitter import Event
from granite_core.events import (
    GeneratingCitationsCompleteEvent,
    GeneratingCitationsEvent,
    PassThroughEvent,
    TextEvent,
    ThinkEvent,
    TrajectoryEvent,
)
from granite_core.gurardrails.copyright import CopyrightViolationGuardrail
from granite_core.logging import get_logger
from granite_core.memory import estimate_tokens, exceeds_token_limit, token_limit_response
from granite_core.research.researcher import Researcher
from granite_core.search.embeddings.tokenizer import EmbeddingsTokenizer
from granite_core.search.prompts import SearchPrompts
from granite_core.search.tool import SearchTool
from granite_core.thinking.prompts import ThinkingPrompts
from granite_core.thinking.response_parser import ThinkingResponseParser
from granite_core.thinking.stream_handler import TagStartEvent, ThinkingStreamHandler, TokenEvent
from granite_core.thinking.tool import ThinkingTool
from granite_core.usage import create_usage_info
from granite_core.utils import log_settings
from granite_core.work import chat_pool
from langchain_core.documents import Document
from redis.asyncio import Redis

from acp_agent import utils
from acp_agent.config import settings
from acp_agent.heartbeat import Heartbeat
from acp_agent.phases import GeneratingCitationsPhase, SearchingWebPhase, Status
from acp_agent.resources import AsyncCachingResourceLoader, ResourceStoreFactory
from acp_agent.store import PrefixRouterMemoryStore

logger = get_logger(__name__)
log_settings(settings, name="Agent")
log_settings(core_settings)

# Preload the embeddings tokenizer if set
EmbeddingsTokenizer.get_instance()

server = Server()


def log_context(input: list[Message], context: Context) -> None:
    logger.info(f">>> Session ID: {context.session.id}")
    if input:
        logger.info(f">>> Message: {input[-1]}")


@server.agent(
    name="granite-chat",
    description="Uses Granite models for general chat.",
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
        recommended_models=["ibm-granite/granite-4.0-h-small", "ibm-granite/granite-3.3-8b-instruct"],
    ),  # type: ignore[call-arg]
)
async def granite_chat(input: list[Message], context: Context) -> AsyncGenerator:
    hb = Heartbeat(context=context, interval=settings.HEARTBEAT_INTERVAL)
    hb.start()

    try:
        log_context(input, context)

        history = [message async for message in context.session.load_history()]
        messages = utils.to_beeai_framework_messages(messages=history + input)

        chat_model = ChatModelFactory.create()

        guardrail = CopyrightViolationGuardrail(chat_model=chat_model)
        guardrail_result = await guardrail.evaluate(messages)

        if guardrail_result.is_harmful:
            messages = [
                SystemMessage(
                    content=f"Providing an answer to the user would result in a potential copyright violation.\nReason: {guardrail_result.reason}\n\nInform the user and suggest alternatives."  # noqa: E501
                ),
                *messages,
            ]
        else:
            messages = [SystemMessage(content=ChatPrompts.chat_system_prompt()), *messages]

        token_count = estimate_tokens(messages=messages)
        if exceeds_token_limit(token_count):
            yield token_limit_response(token_count)
            return

        if core_settings.STREAMING is True:
            async with chat_pool.throttle():
                async for event, _ in chat_model.run(messages, max_retries=core_settings.MAX_RETRIES, stream=True):
                    if isinstance(event, ChatModelNewTokenEvent):
                        yield MessagePart(content=event.value.get_text_content())
                    elif isinstance(event, ChatModelSuccessEvent):
                        yield create_usage_info(event.value.usage, chat_model.model_id)
        else:
            async with chat_pool.throttle():
                output = await chat_model.run(messages, max_retries=core_settings.MAX_RETRIES)
            yield MessagePart(content=output.get_text_content())
            yield create_usage_info(output.usage, chat_model.model_id)

    except Exception as e:
        logger.exception(repr(e))
        raise e
    finally:
        await hb.stop()


@server.agent(
    name="granite-thinking",
    description="Uses Granite models for general chat with reasoning.",
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
    ),  # type: ignore[call-arg]
)
async def granite_think(input: list[Message], context: Context) -> AsyncGenerator:
    hb = Heartbeat(context=context, interval=settings.HEARTBEAT_INTERVAL)
    hb.start()

    try:
        log_context(input, context)

        history = [message async for message in context.session.load_history()]
        messages = utils.to_beeai_framework_messages(messages=history + input)

        chat_model = ChatModelFactory.create()

        guardrail = CopyrightViolationGuardrail(chat_model=chat_model)
        guardrail_result = await guardrail.evaluate(messages)

        if guardrail_result.is_harmful:
            messages = [
                SystemMessage(
                    content=f"Providing an answer to the user would result in a potential copyright violation.\nReason: {guardrail_result.reason}\n\nInform the user and suggest alternatives."  # noqa: E501
                ),
                *messages,
            ]

        token_count = estimate_tokens(messages=messages)
        if exceeds_token_limit(token_count):
            yield token_limit_response(token_count)
            return

        if settings.TWO_STEP_THINKING is True:

            async def think_handler(event: Event) -> None:
                if isinstance(event, ThinkEvent):
                    await context.yield_async(MessagePart(content_type="text/thinking", content=event.text))
                if isinstance(event, TextEvent):
                    await context.yield_async(MessagePart(content=event.text))
                elif isinstance(event, PassThroughEvent) and isinstance(event.event, ChatModelSuccessEvent):
                    await context.yield_async(
                        create_usage_info(cast(ChatModelSuccessEvent, event.event).value.usage, chat_model.model_id)
                    )

            think_tool = ThinkingTool(
                chat_model=chat_model,
                messages=messages,
                session_id=str(context.session.id),
            )
            think_tool.subscribe(handler=think_handler)
            await think_tool.run()
        else:
            messages = [
                SystemMessage(content=ThinkingPrompts.granite3_3_thinking_system_prompt()),
                *messages,
            ]

            handler = ThinkingStreamHandler(tags=["think", "response"])

            if core_settings.STREAMING is True:
                async with chat_pool.throttle():
                    async for event, _ in chat_model.run(messages, stream=True, max_retries=core_settings.MAX_RETRIES):
                        if isinstance(event, ChatModelNewTokenEvent):
                            token = event.value.get_text_content()
                            for output in handler.on_token(token=token):
                                if isinstance(output, TokenEvent):
                                    content_type = "text/thinking" if output.tag == "think" else "text/plain"
                                    yield MessagePart(content_type=content_type, content=output.token)
                                elif isinstance(output, TagStartEvent):
                                    pass
                        elif isinstance(event, ChatModelSuccessEvent):
                            yield create_usage_info(event.value.usage, chat_model.model_id)
            else:
                async with chat_pool.throttle():
                    chat_output = await chat_model.run(messages, max_retries=core_settings.MAX_RETRIES)

                text = chat_output.get_text_content()

                parser = ThinkingResponseParser()
                think_resp = parser.parse(text=text)

                if think_resp.thinking is not None:
                    yield MessagePart(content_type="text/thinking", content=think_resp.thinking)

                if think_resp.response is not None:
                    yield MessagePart(content_type="text/plain", content=think_resp.response)

                yield create_usage_info(chat_output.usage, chat_model.model_id)

    except Exception as e:
        logger.exception(repr(e))
        raise e
    finally:
        await hb.stop()


@server.agent(
    name="granite-search",
    description="Uses Granite models to chat and search the web.",
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
        recommended_models=["ibm-granite/granite-4.0-h-small", "ibm-granite/granite-3.3-8b-instruct"],
    ),  # type: ignore[call-arg]
)
async def granite_search(input: list[Message], context: Context) -> AsyncGenerator:
    hb = Heartbeat(context=context, interval=settings.HEARTBEAT_INTERVAL)
    hb.start()

    try:
        log_context(input, context)

        history = [message async for message in context.session.load_history()]
        messages = utils.to_beeai_framework_messages(messages=history + input)

        chat_model = ChatModelFactory.create()
        structured_chat_model = ChatModelFactory.create(model_type="structured")

        guardrail = CopyrightViolationGuardrail(chat_model=chat_model)
        guardrail_result = await guardrail.evaluate(messages)

        if guardrail_result.is_harmful:
            messages = [
                SystemMessage(
                    content=f"Providing an answer to the user would result in a potential copyright violation.\nReason: {guardrail_result.reason}\n\nInform the user and suggest alternatives."  # noqa: E501
                ),
                *messages,
            ]

        token_count = estimate_tokens(messages=messages)
        if exceeds_token_limit(token_count):
            yield token_limit_response(token_count)
            return

        response: list[str] = []

        if guardrail_result.is_harmful:
            if core_settings.STREAMING is True:
                async with chat_pool.throttle():
                    async for event, _ in chat_model.run(messages, stream=True, max_retries=core_settings.MAX_RETRIES):
                        if isinstance(event, ChatModelNewTokenEvent):
                            content = event.value.get_text_content()
                            response.append(content)
                            yield MessagePart(content=content)
                        elif isinstance(event, ChatModelSuccessEvent):
                            yield create_usage_info(event.value.usage, chat_model.model_id)
            else:
                async with chat_pool.throttle():
                    output = await chat_model.run(messages, max_retries=core_settings.MAX_RETRIES)

                response.append(output.get_text_content())
                yield MessagePart(content="".join(response))
                yield create_usage_info(output.usage, chat_model.model_id)

            return

        await context.yield_async(SearchingWebPhase(status=Status.active).wrapped)

        search_tool = SearchTool(chat_model=structured_chat_model, session_id=str(context.session.id))
        docs: list[Document] = await search_tool.search(messages)

        if len(docs) > 0:
            doc_messages: list[FrameworkMessage] = [SystemMessage(content=SearchPrompts.search_system_prompt(docs))]
            # Prepend document prompt
            messages = doc_messages + messages
        else:
            messages = [SystemMessage(content=ChatPrompts.chat_system_prompt()), *messages]

        await context.yield_async(SearchingWebPhase(status=Status.completed).wrapped)

        if core_settings.STREAMING is True:
            async with chat_pool.throttle():
                async for event, _ in chat_model.run(messages, stream=True, max_retries=core_settings.MAX_RETRIES):
                    if isinstance(event, ChatModelNewTokenEvent):
                        content = event.value.get_text_content()
                        response.append(content)
                        yield MessagePart(content=content)
                    elif isinstance(event, ChatModelSuccessEvent):
                        yield create_usage_info(event.value.usage, chat_model.model_id)
        else:
            async with chat_pool.throttle():
                output = await chat_model.run(messages, max_retries=core_settings.MAX_RETRIES)

            response.append(output.get_text_content())
            yield MessagePart(content="".join(response))
            yield create_usage_info(output.usage, chat_model.model_id)

        # Yield sources/citation
        if len(docs) > 0:

            async def citation_handler(event: Event) -> None:
                if isinstance(event, CitationEvent):
                    logger.info(f"Citation: {event.citation.url}")
                    await context.yield_async(utils.to_citation_message_part(event.citation))

            generator = CitationGeneratorFactory.create()
            generator.subscribe(handler=citation_handler)

            await context.yield_async(GeneratingCitationsPhase(status=Status.active).wrapped)
            await generator.generate(docs=docs, response="".join(response))
            await context.yield_async(GeneratingCitationsPhase(status=Status.completed).wrapped)

    except Exception as e:
        logger.exception(repr(e))
        raise e
    finally:
        await hb.stop()


@server.agent(
    name="granite-research",
    description="Uses Granite models to perform research.",
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
        recommended_models=["ibm-granite/granite-4.0-h-small", "ibm-granite/granite-3.3-8b-instruct"],
    ),  # type: ignore[call-arg]
)
async def granite_research(input: list[Message], context: Context) -> AsyncGenerator:
    hb = Heartbeat(context=context, interval=settings.HEARTBEAT_INTERVAL)
    hb.start()

    try:
        log_context(input, context)

        history = [message async for message in context.session.load_history()]
        messages = utils.to_beeai_framework_messages(messages=history + input)

        token_count = estimate_tokens(messages=messages)
        if exceeds_token_limit(token_count):
            yield token_limit_response(token_count)
            return

        chat_model = ChatModelFactory.create()
        structured_chat_model = ChatModelFactory.create(model_type="structured")

        async def research_listener(event: Event) -> None:
            if isinstance(event, TextEvent):
                await context.yield_async(MessagePart(content=event.text))
            elif isinstance(event, PassThroughEvent) and isinstance(event.event, ChatModelSuccessEvent):
                await context.yield_async(
                    create_usage_info(cast(ChatModelSuccessEvent, event.event).value.usage, chat_model.model_id)
                )
            elif isinstance(event, TrajectoryEvent):
                await context.yield_async(
                    MessagePart(content="", metadata=TrajectoryMetadata(message=event.to_markdown()))
                )
            elif isinstance(event, GeneratingCitationsEvent):
                await context.yield_async(GeneratingCitationsPhase(status=Status.active).wrapped)
            elif isinstance(event, CitationEvent):
                logger.info(f"[granite_research:{context.session.id}] Citation: {event.citation.url}")
                await context.yield_async(utils.to_citation_message_part(event.citation))
            elif isinstance(event, GeneratingCitationsCompleteEvent):
                await context.yield_async(GeneratingCitationsPhase(status=Status.completed).wrapped)

        researcher = Researcher(
            chat_model=chat_model,
            structured_chat_model=structured_chat_model,
            messages=messages,
            session_id=str(context.session.id),
        )
        researcher.subscribe(handler=research_listener)
        await researcher.run()

    except Exception as e:
        logger.exception(repr(e))
        raise e
    finally:
        await hb.stop()


store: PrefixRouterMemoryStore = PrefixRouterMemoryStore(debounce=settings.MEM_STORE_NOTIFICATION_DEBOUNCE)

if settings.KEY_STORE_PROVIDER == "redis" and settings.REDIS_CLIENT_URL is not None:
    logger.info("Found a valid redis KEY_STORE_PROVIDER")
    redis = Redis().from_url(settings.REDIS_CLIENT_URL.get_secret_value())
    # Sessions are stored in persistent store, everything else to memory
    store.map_prefix("session_", RedisStore(redis=redis))

resource_store = ResourceStoreFactory.create()
forward_resources = resource_store is None
resource_loader = AsyncCachingResourceLoader()

server.run(
    configure_logger=False,
    host=settings.HOST,
    port=settings.PORT,
    access_log=settings.ACCESS_LOG,
    store=store,
    resource_store=resource_store,
    forward_resources=forward_resources,
    resource_loader=resource_loader,
)
