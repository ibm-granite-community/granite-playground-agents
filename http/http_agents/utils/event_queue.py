# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0


import asyncio
from collections.abc import AsyncGenerator

from granite_core.emitter import Event


class EventStreamQueue:
    """Queue that bridges event handlers with async generators for real-time streaming"""

    def __init__(self) -> None:
        self._queue: asyncio.Queue[Event | None] = asyncio.Queue()

    async def handler(self, event: Event) -> None:
        """Event handler that adds events to queue - subscribe this to EventEmitter"""
        await self._queue.put(event)

    async def stream(self) -> AsyncGenerator[Event, None]:
        """Stream events from queue in real-time as they arrive"""
        while True:
            event = await self._queue.get()
            if event is None:  # Sentinel to stop
                break
            yield event

    async def stop(self) -> None:
        """Signal the stream to stop by putting sentinel value"""
        await self._queue.put(None)


# Made with Bob
