# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0


import asyncio
from datetime import UTC

from acp_sdk import Field, datetime
from acp_sdk.server import Context
from pydantic import BaseModel


class HeartBeatMessage(BaseModel):
    type: str = "heartbeat"
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class Heartbeat:
    def __init__(self, context: Context, interval: float = 10) -> None:
        self._context = context
        self._interval = interval
        self._stop_event = asyncio.Event()
        self._task: asyncio.Task | None = None

    def start(self) -> None:
        """Start the heartbeat loop in the background."""
        if self._task is None or self._task.done():
            self._stop_event.clear()
            self._task = asyncio.create_task(self._run())

    async def _run(self) -> None:
        while not self._stop_event.is_set():
            await self._context.yield_async(HeartBeatMessage())
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=self._interval)
            except TimeoutError:
                continue

    async def stop(self) -> None:
        """Stop the heartbeat loop and wait for it to finish."""
        self._stop_event.set()
        if self._task:
            await self._task
