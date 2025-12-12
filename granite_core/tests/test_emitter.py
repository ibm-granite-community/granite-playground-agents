# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0

from datetime import UTC, datetime
from typing import TypeVar
from unittest.mock import AsyncMock

import pytest

from granite_core.emitter import Event, EventEmitter, EventHandler
from granite_core.events import TrajectoryEvent

T = TypeVar("T", bound=Event)


def test_event_timestamp() -> None:
    event = Event()
    assert isinstance(event.timestamp, datetime)
    assert event.timestamp.tzinfo == UTC


def test_trajectory_event_markdown() -> None:
    event = TrajectoryEvent(title="Test")
    assert "**Test**" in event.to_markdown()

    event = TrajectoryEvent(title="Test", content=["Trajectory", "String"])
    assert all(s in event.to_markdown() for s in ["**Test**", "- Trajectory\n", "- String"])

    event = TrajectoryEvent(title="Test", content="Trajectory String")
    assert all(s in event.to_markdown() for s in ["**Test**", "Trajectory String"])


@pytest.fixture
def event_emitter() -> EventEmitter:
    return EventEmitter()


@pytest.fixture
def event_handler() -> AsyncMock:
    return AsyncMock()


def test_subscribe(event_emitter: EventEmitter, event_handler: EventHandler) -> None:
    event_emitter.subscribe(event_handler)
    assert event_handler in event_emitter._handlers


def test_unsubscribe(event_emitter: EventEmitter, event_handler: EventHandler) -> None:
    event_emitter.subscribe(event_handler)
    event_emitter.unsubscribe(event_handler)
    assert event_handler not in event_emitter._handlers


@pytest.mark.asyncio
async def test_emit(event_emitter: EventEmitter, event_handler: AsyncMock) -> None:
    event = Event()
    event_emitter.subscribe(event_handler)
    await event_emitter._emit(event)
    event_handler.assert_called_once_with(event)


@pytest.mark.asyncio
async def test_forward_events_from(event_emitter: EventEmitter, event_handler: AsyncMock) -> None:
    source_emitter = EventEmitter()
    event = Event()
    event_emitter.subscribe(event_handler)
    event_emitter.forward_events_from(source_emitter)
    await source_emitter._emit(event)
    event_handler.assert_called_once_with(event)
