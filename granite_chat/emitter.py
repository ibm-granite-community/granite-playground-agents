import asyncio
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime

from pydantic import BaseModel, Field


class Event(BaseModel):
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    type: str
    data: str


LogEventHandler = Callable[[Event], Awaitable[None]]


class EventEmitter:
    def __init__(self) -> None:
        self._handlers: list[LogEventHandler] = []

    def subscribe(self, handler: LogEventHandler) -> None:
        """Register a handler to receive log events"""
        self._handlers.append(handler)

    def unsubscribe(self, handler: LogEventHandler) -> None:
        """Remove a previously registered handler"""
        self._handlers.remove(handler)

    async def _emit(self, event: Event) -> None:
        """Emit to all handlers concurrently"""
        await asyncio.gather(*(handler(event) for handler in self._handlers))

    def forward_events_from(self, other: "EventEmitter") -> None:
        """Subscribe to another emitter and forward its events to this emitter's subscribers"""

        async def _forward(event: Event) -> None:
            await self._emit(event)

        other.subscribe(_forward)
