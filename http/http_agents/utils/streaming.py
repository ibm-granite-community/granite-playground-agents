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


async def send_heartbeat(interval: float = 30.0) -> AsyncGenerator[str, None]:
    """Send periodic heartbeat events."""
    while True:
        await asyncio.sleep(interval)
        yield await send_sse_event("heartbeat", {"timestamp": asyncio.get_event_loop().time()})


async def stream_with_heartbeat(
    content_generator: AsyncGenerator[str, None], heartbeat_interval: float = 10.0
) -> AsyncGenerator[str, None]:
    """
    Combine content stream with heartbeat events.

    Uses a queue to merge content events and periodic heartbeats into a single stream.
    This ensures heartbeats are sent at regular intervals even during idle periods.

    Args:
        content_generator: Async generator yielding content
        heartbeat_interval: Seconds between heartbeat events
    """
    queue: asyncio.Queue[str | None] = asyncio.Queue()
    content_task = None
    heartbeat_task = None

    async def content_reader() -> None:
        """Read content from generator and put in queue."""
        try:
            async for content in content_generator:
                await queue.put(content)
        finally:
            await queue.put(None)  # Signal completion

    async def heartbeat_sender() -> None:
        """Send heartbeats at regular intervals."""
        try:
            while True:
                await asyncio.sleep(heartbeat_interval)
                await queue.put(await send_sse_event("heartbeat", {"timestamp": asyncio.get_event_loop().time()}))
        except asyncio.CancelledError:
            pass

    try:
        # Start both tasks
        content_task = asyncio.create_task(content_reader())
        heartbeat_task = asyncio.create_task(heartbeat_sender())

        # Yield from queue until content is done
        while True:
            item = await queue.get()
            if item is None:  # Content finished
                break
            yield item

    finally:
        # Cleanup tasks
        if heartbeat_task:
            heartbeat_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await heartbeat_task
        if content_task and not content_task.done():
            content_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await content_task
