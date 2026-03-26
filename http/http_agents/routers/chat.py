# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0


from collections.abc import AsyncGenerator
from typing import cast

from beeai_framework.backend import ChatModelSuccessEvent
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from granite_core.chat.handler import ChatHandler
from granite_core.chat_model import ChatModelFactory
from granite_core.config import settings as core_settings
from granite_core.emitter import Event
from granite_core.events import PassThroughEvent, TextEvent, TokenLimitExceededEvent
from granite_core.logging import get_logger
from granite_core.memory import estimate_tokens, exceeds_token_limit
from granite_core.usage import create_usage_info
from granite_core.work import chat_pool

from http_agents.config import settings
from http_agents.models.requests import ChatRequest
from http_agents.models.responses import ChatResponse, ErrorResponse
from http_agents.services.session_manager import session_manager
from http_agents.utils.converters import convert_usage_info
from http_agents.utils.streaming import send_sse_event, stream_with_heartbeat

logger = get_logger(__name__)
router = APIRouter(prefix="/chat", tags=["chat"])


@router.post(
    "",
    response_model=ChatResponse,
    responses={
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
    summary="Chat with Granite model",
    description="Send a message and receive a response from the Granite chat model without external tools.",
)
async def chat(request: ChatRequest) -> StreamingResponse | ChatResponse:
    """
    Chat endpoint for basic conversation without external tools.

    Supports both streaming and non-streaming responses.
    """
    try:
        logger.info(f"Chat request for session {request.session_id}")

        # Add user message to history
        await session_manager.add_message(request.session_id, "user", request.message)

        # Get conversation history
        messages = await session_manager.get_framework_messages(request.session_id)

        # Check token limit
        token_count = estimate_tokens(messages=messages)
        if exceeds_token_limit(token_count):
            raise HTTPException(
                status_code=400,
                detail=f"Token limit exceeded. Estimated tokens: {token_count}, limit: {core_settings.CHAT_TOKEN_LIMIT}",  # noqa: E501
            )

        # Create chat model
        chat_model = ChatModelFactory.create()

        if request.stream:
            # Streaming response
            async def generate_stream() -> AsyncGenerator[str, None]:
                agent_response_text: list[str] = []
                usage_info = None
                event_queue: list[str] = []

                async def chat_listener(event: Event) -> None:
                    nonlocal usage_info
                    if isinstance(event, TextEvent):
                        agent_response_text.append(event.text)
                        event_queue.append(await send_sse_event("token", {"content": event.text}))
                    elif isinstance(event, TokenLimitExceededEvent):
                        event_queue.append(
                            await send_sse_event(
                                "error", {"message": f"Token limit exceeded: {event.estimated_tokens} tokens"}
                            )
                        )
                    elif isinstance(event, PassThroughEvent) and isinstance(event.event, ChatModelSuccessEvent):
                        usage_info = create_usage_info(
                            usage=cast(ChatModelSuccessEvent, event.event).value.usage, model_id=chat_model.model_id
                        )

                chat_handler = ChatHandler(
                    chat_model=chat_model, session_id=request.session_id, token_limit=core_settings.CHAT_TOKEN_LIMIT
                )
                chat_handler.subscribe(handler=chat_listener)

                async with chat_pool.throttle():
                    await chat_handler.run(messages, stream=core_settings.STREAMING)

                # Yield all queued events
                for event_data in event_queue:
                    yield event_data

                # Send usage info
                if usage_info:
                    yield await send_sse_event("usage", convert_usage_info(usage_info).model_dump())

                # Send done event
                yield await send_sse_event("done", {"session_id": request.session_id})

                # Save assistant response to history
                full_response = "".join(agent_response_text)
                await session_manager.add_message(request.session_id, "assistant", full_response)
                logger.info(f"Chat response for session {request.session_id}: {full_response[:100]}...")

            return StreamingResponse(
                stream_with_heartbeat(generate_stream(), settings.HEARTBEAT_INTERVAL), media_type="text/event-stream"
            )

        else:
            # Non-streaming response
            agent_response_text: list[str] = []
            usage_info = None

            async def chat_listener(event: Event) -> None:
                nonlocal usage_info
                if isinstance(event, TextEvent):
                    agent_response_text.append(event.text)
                elif isinstance(event, TokenLimitExceededEvent):
                    raise HTTPException(
                        status_code=400, detail=f"Token limit exceeded: {event.estimated_tokens} tokens"
                    )
                elif isinstance(event, PassThroughEvent) and isinstance(event.event, ChatModelSuccessEvent):
                    usage_info = create_usage_info(
                        usage=cast(ChatModelSuccessEvent, event.event).value.usage, model_id=chat_model.model_id
                    )

            chat_handler = ChatHandler(
                chat_model=chat_model, session_id=request.session_id, token_limit=core_settings.CHAT_TOKEN_LIMIT
            )
            chat_handler.subscribe(handler=chat_listener)

            async with chat_pool.throttle():
                await chat_handler.run(messages, stream=False)

            full_response = "".join(agent_response_text)
            await session_manager.add_message(request.session_id, "assistant", full_response)

            logger.info(f"Chat response for session {request.session_id}: {full_response[:100]}...")

            return ChatResponse(
                response=full_response,
                usage=convert_usage_info(usage_info) if usage_info else None,
                session_id=request.session_id,
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e
