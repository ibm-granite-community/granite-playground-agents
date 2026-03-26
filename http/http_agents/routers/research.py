# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0


from collections.abc import AsyncGenerator
from typing import cast

from beeai_framework.backend import ChatModelSuccessEvent
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from granite_core.chat_model import ChatModelFactory
from granite_core.citations.events import CitationEvent
from granite_core.emitter import Event
from granite_core.events import (
    GeneratingCitationsCompleteEvent,
    GeneratingCitationsEvent,
    PassThroughEvent,
    TextEvent,
    TrajectoryEvent,
)
from granite_core.logging import get_logger
from granite_core.memory import estimate_tokens, exceeds_token_limit
from granite_core.research.researcher import Researcher
from granite_core.usage import create_usage_info as create_granite_usage_info

from http_agents.config import settings
from http_agents.models.requests import ResearchRequest
from http_agents.models.responses import Citation, Phase, ResearchResponse
from http_agents.services.session_manager import session_manager
from http_agents.utils.converters import convert_citation, convert_usage_info
from http_agents.utils.streaming import send_sse_event, stream_with_heartbeat

logger = get_logger(__name__)
router = APIRouter(prefix="/research", tags=["research"])


@router.post(
    "",
    response_model=ResearchResponse,
    summary="Deep research with Granite model",
    description="Perform deep research on a topic with iterative web search and detailed analysis.",
)
async def research(request: ResearchRequest) -> StreamingResponse | ResearchResponse:
    """
    Research endpoint for deep research with iterative search and analysis.
    """
    try:
        logger.info(f"Research request for session {request.session_id}")

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

        if request.stream:
            # Streaming response
            async def generate_stream() -> AsyncGenerator[str, None]:
                response_text: list[str] = []
                citations_list: list[dict] = []
                trajectory_list: list[str] = []
                phases: list[dict] = []
                usage_info = None
                event_queue: list[str] = []

                async def queuing_listener(event: Event) -> None:
                    nonlocal usage_info
                    if isinstance(event, TextEvent):
                        response_text.append(event.text)
                        event_queue.append(await send_sse_event("token", {"content": event.text}))
                    elif isinstance(event, PassThroughEvent) and isinstance(event.event, ChatModelSuccessEvent):
                        usage_info = create_granite_usage_info(
                            cast(ChatModelSuccessEvent, event.event).value.usage, chat_model.model_id
                        )
                    elif isinstance(event, TrajectoryEvent):
                        trajectory_md = event.to_markdown()
                        trajectory_list.append(trajectory_md)
                        event_queue.append(await send_sse_event("trajectory", {"message": trajectory_md}))
                    elif isinstance(event, GeneratingCitationsEvent):
                        event_queue.append(
                            await send_sse_event("phase", {"name": "generating-citations", "status": "active"})
                        )
                        phases.append({"name": "generating-citations", "status": "active"})
                    elif isinstance(event, CitationEvent):
                        citation_dict = convert_citation(event.citation).model_dump()
                        citations_list.append(citation_dict)
                        event_queue.append(await send_sse_event("citation", citation_dict))
                    elif isinstance(event, GeneratingCitationsCompleteEvent):
                        event_queue.append(
                            await send_sse_event("phase", {"name": "generating-citations", "status": "completed"})
                        )
                        phases.append({"name": "generating-citations", "status": "completed"})

                researcher = Researcher(
                    chat_model=chat_model,
                    structured_chat_model=structured_chat_model,
                    messages=messages,
                    session_id=request.session_id,
                )

                researcher.subscribe(handler=queuing_listener)
                await researcher.run()

                # Yield all queued events
                for event_data in event_queue:
                    yield event_data

                # Send usage info
                if usage_info:
                    yield await send_sse_event("usage", convert_usage_info(usage_info).model_dump())

                # Send done event
                yield await send_sse_event("done", {"session_id": request.session_id})

                # Save assistant response
                full_response = "".join(response_text)
                await session_manager.add_message(request.session_id, "assistant", full_response)
                logger.info(f"Research response for session {request.session_id}: {full_response[:100]}...")

            return StreamingResponse(
                stream_with_heartbeat(generate_stream(), settings.HEARTBEAT_INTERVAL), media_type="text/event-stream"
            )

        else:
            # Non-streaming response
            response_text: list[str] = []
            citations_list: list[Citation] = []
            trajectory_list: list[str] = []
            phases: list[Phase] = []
            usage_info = None

            async def research_listener(event: Event) -> None:
                nonlocal usage_info
                if isinstance(event, TextEvent):
                    response_text.append(event.text)
                elif isinstance(event, PassThroughEvent) and isinstance(event.event, ChatModelSuccessEvent):
                    usage_info = create_granite_usage_info(
                        cast(ChatModelSuccessEvent, event.event).value.usage, chat_model.model_id
                    )
                elif isinstance(event, TrajectoryEvent):
                    trajectory_list.append(event.to_markdown())
                elif isinstance(event, GeneratingCitationsEvent):
                    phases.append(Phase(name="generating-citations", status="active"))
                elif isinstance(event, CitationEvent):
                    citations_list.append(convert_citation(event.citation))
                elif isinstance(event, GeneratingCitationsCompleteEvent):
                    phases.append(Phase(name="generating-citations", status="completed"))

            researcher = Researcher(
                chat_model=chat_model,
                structured_chat_model=structured_chat_model,
                messages=messages,
                session_id=request.session_id,
            )
            researcher.subscribe(handler=research_listener)
            await researcher.run()

            full_response = "".join(response_text)
            await session_manager.add_message(request.session_id, "assistant", full_response)

            logger.info(f"Research response for session {request.session_id}: {full_response[:100]}...")

            return ResearchResponse(
                response=full_response,
                citations=citations_list,
                trajectory=trajectory_list,
                phases=phases,
                usage=convert_usage_info(usage_info) if usage_info else None,
                session_id=request.session_id,
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error in research endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e
