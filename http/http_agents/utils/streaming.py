# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0


import asyncio
import contextlib
from collections.abc import AsyncGenerator
from typing import Any

from granite_core.logging import get_logger

from http_agents.models.responses import StreamEvent

logger = get_logger(__name__)


async def send_sse_event(event_type: str, data: Any) -> str:
    """Format data as Server-Sent Event."""
    event = StreamEvent(type=event_type, data=data)  # type: ignore
    return f"data: {event.model_dump_json()}\n\n"


async def send_heartbeat(interval: float = 10.0) -> AsyncGenerator[str, None]:
    """Send periodic heartbeat events."""
    while True:
        await asyncio.sleep(interval)
        yield await send_sse_event("heartbeat", {"timestamp": asyncio.get_event_loop().time()})


async def stream_with_heartbeat(
    content_generator: AsyncGenerator[str, None], heartbeat_interval: float = 10.0
) -> AsyncGenerator[str, None]:
    """
    Combine content stream with heartbeat events.

    Args:
        content_generator: Async generator yielding content
        heartbeat_interval: Seconds between heartbeat events
    """
    heartbeat_task = None
    heartbeat_queue: asyncio.Queue = asyncio.Queue()

    async def heartbeat_sender() -> None:
        """Background task to send heartbeats."""
        try:
            while True:
                await asyncio.sleep(heartbeat_interval)
                await heartbeat_queue.put(
                    await send_sse_event("heartbeat", {"timestamp": asyncio.get_event_loop().time()})
                )
        except asyncio.CancelledError:
            pass

    try:
        # Start heartbeat task
        heartbeat_task = asyncio.create_task(heartbeat_sender())

        # Yield content and heartbeats
        async for content in content_generator:
            # Check for any pending heartbeats
            while not heartbeat_queue.empty():
                try:
                    heartbeat = heartbeat_queue.get_nowait()
                    yield heartbeat
                except asyncio.QueueEmpty:
                    break

            # Yield the actual content
            yield content

    finally:
        # Cleanup heartbeat task
        if heartbeat_task:
            heartbeat_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await heartbeat_task
