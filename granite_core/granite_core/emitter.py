# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0


import asyncio
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any, TypeVar

from pydantic import BaseModel, Field


class Event(BaseModel):
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


T = TypeVar("T", bound=Event)


EventHandler = Callable[[T], Awaitable[None]]


class EventEmitter:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._handlers: list[EventHandler] = []
        super().__init__(*args, **kwargs)

    def subscribe(self, handler: EventHandler) -> None:
        """Register a handler to receive events"""
        self._handlers.append(handler)

    def unsubscribe(self, handler: EventHandler) -> None:
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
