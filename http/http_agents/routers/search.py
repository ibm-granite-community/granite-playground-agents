# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0


from collections.abc import AsyncGenerator

from beeai_framework.backend import ChatModelNewTokenEvent, ChatModelSuccessEvent, SystemMessage
from beeai_framework.backend import Message as FrameworkMessage
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from granite_core.chat.prompts import ChatPrompts
from granite_core.chat_model import ChatModelFactory
from granite_core.citations.citations import CitationGeneratorFactory
from granite_core.citations.events import CitationEvent
from granite_core.config import settings as core_settings
from granite_core.emitter import Event
from granite_core.gurardrails.copyright import CopyrightViolationGuardrail
from granite_core.logging import get_logger
from granite_core.memory import estimate_tokens, exceeds_token_limit
from granite_core.search.prompts import SearchPrompts
from granite_core.search.tool import SearchTool
from granite_core.usage import create_usage_info
from granite_core.work import chat_pool
from langchain_core.documents import Document

from http_agents.config import settings
from http_agents.models.requests import SearchRequest
from http_agents.models.responses import Citation, Phase, SearchResponse
from http_agents.services.session_manager import session_manager
from http_agents.utils.converters import convert_citation, convert_usage_info
from http_agents.utils.streaming import send_sse_event, stream_with_heartbeat

logger = get_logger(__name__)
router = APIRouter(prefix="/search", tags=["search"])


@router.post(
    "",
    response_model=SearchResponse,
    summary="Search and chat with Granite model",
    description="Send a query, search the web, and receive a response with citations.",
)
async def search(request: SearchRequest) -> StreamingResponse | SearchResponse:
    """
    Search endpoint for web-augmented chat with citations.
    """
    try:
        logger.info(f"Search request for session {request.session_id}")

        # Add user message to history
        await session_manager.add_message(request.session_id, "user", request.message)

        # Get conversation history
        messages = await session_manager.get_framework_messages(request.session_id)

        # Check token limit
        token_count = estimate_tokens(messages=messages)
        if exceeds_token_limit(token_count):
            raise HTTPException(status_code=400, detail=f"Token limit exceeded. Estimated tokens: {token_count}")

        # Create chat models
        chat_model = ChatModelFactory.create()
        structured_chat_model = ChatModelFactory.create(model_type="structured")

        # Check copyright guardrail
        guardrail = CopyrightViolationGuardrail(chat_model=chat_model)
        guardrail_result = await guardrail.evaluate(messages)

        if guardrail_result.violated:
            messages = [
                SystemMessage(
                    content=f"Providing an answer to the user would result in a potential copyright violation.\nReason: {guardrail_result.reason}\n\nInform the user and suggest alternatives."  # noqa: E501
                ),
                *messages,
            ]

        if request.stream:
            # Streaming response
            async def generate_stream() -> AsyncGenerator[str, None]:
                nonlocal messages
                response_text: list[str] = []
                citations_list: list[dict] = []
                phases: list[dict] = []
                usage_info = None
                docs: list[Document] = []

                # Phase: Searching web
                if not guardrail_result.violated:
                    yield await send_sse_event("phase", {"name": "searching-web", "status": "active"})
                    phases.append({"name": "searching-web", "status": "active"})

                    search_tool = SearchTool(chat_model=structured_chat_model, session_id=request.session_id)
                    docs = await search_tool.search(messages)

                    if len(docs) > 0:
                        doc_messages: list[FrameworkMessage] = [
                            SystemMessage(content=SearchPrompts.search_system_prompt(docs))
                        ]
                        messages = doc_messages + messages
                    else:
                        messages = [SystemMessage(content=ChatPrompts.chat_system_prompt()), *messages]

                    yield await send_sse_event("phase", {"name": "searching-web", "status": "completed"})
                    phases.append({"name": "searching-web", "status": "completed"})

                # Generate response
                async with chat_pool.throttle():
                    async for event, _ in chat_model.run(messages, stream=True, max_retries=core_settings.MAX_RETRIES):
                        if isinstance(event, ChatModelNewTokenEvent):
                            content = event.value.get_text_content()
                            response_text.append(content)
                            yield await send_sse_event("token", {"content": content})
                        elif isinstance(event, ChatModelSuccessEvent):
                            usage_info = create_usage_info(event.value.usage, chat_model.model_id)

                # Generate citations
                if not guardrail_result.violated and len(docs) > 0:
                    yield await send_sse_event("phase", {"name": "generating-citations", "status": "active"})
                    phases.append({"name": "generating-citations", "status": "active"})

                    citation_events: list[str] = []

                    async def citation_handler(event: Event) -> None:
                        if isinstance(event, CitationEvent):
                            citation_dict = convert_citation(event.citation).model_dump()
                            citations_list.append(citation_dict)
                            citation_events.append(await send_sse_event("citation", citation_dict))

                    generator = CitationGeneratorFactory.create()
                    generator.subscribe(handler=citation_handler)
                    await generator.generate(docs=docs, response="".join(response_text))

                    # Yield citation events
                    for citation_event in citation_events:
                        yield citation_event

                    yield await send_sse_event("phase", {"name": "generating-citations", "status": "completed"})
                    phases.append({"name": "generating-citations", "status": "completed"})

                # Send usage info
                if usage_info:
                    yield await send_sse_event("usage", convert_usage_info(usage_info).model_dump())

                # Send done event
                yield await send_sse_event("done", {"session_id": request.session_id})

                # Save assistant response
                full_response = "".join(response_text)
                await session_manager.add_message(request.session_id, "assistant", full_response)
                logger.info(f"Search response for session {request.session_id}: {full_response[:100]}...")

            return StreamingResponse(
                stream_with_heartbeat(generate_stream(), settings.HEARTBEAT_INTERVAL), media_type="text/event-stream"
            )

        else:
            # Non-streaming response
            response_text: list[str] = []
            citations: list[Citation] = []
            phases: list[Phase] = []
            usage_info = None

            # Search web
            if not guardrail_result.violated:
                phases.append(Phase(name="searching-web", status="active"))

                search_tool = SearchTool(chat_model=structured_chat_model, session_id=request.session_id)
                docs: list[Document] = await search_tool.search(messages)

                if len(docs) > 0:
                    doc_messages: list[FrameworkMessage] = [
                        SystemMessage(content=SearchPrompts.search_system_prompt(docs))
                    ]
                    messages = doc_messages + messages
                else:
                    messages = [SystemMessage(content=ChatPrompts.chat_system_prompt()), *messages]

                phases.append(Phase(name="searching-web", status="completed"))

            # Generate response
            async with chat_pool.throttle():
                output = await chat_model.run(messages, max_retries=core_settings.MAX_RETRIES)

            response_text.append(output.get_text_content())
            usage_info = create_usage_info(output.usage, chat_model.model_id)

            # Generate citations
            if not guardrail_result.violated and len(docs) > 0:
                phases.append(Phase(name="generating-citations", status="active"))

                generator = CitationGeneratorFactory.create()

                async def citation_handler(event: Event) -> None:
                    if isinstance(event, CitationEvent):
                        citations.append(convert_citation(event.citation))

                generator.subscribe(handler=citation_handler)
                await generator.generate(docs=docs, response="".join(response_text))

                phases.append(Phase(name="generating-citations", status="completed"))

            full_response = "".join(response_text)
            await session_manager.add_message(request.session_id, "assistant", full_response)

            logger.info(f"Search response for session {request.session_id}: {full_response[:100]}...")

            return SearchResponse(
                response=full_response,
                citations=citations,
                phases=phases,
                usage=convert_usage_info(usage_info) if usage_info else None,
                session_id=request.session_id,
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error in search endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e
