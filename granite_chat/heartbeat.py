import asyncio

from acp_sdk.server import Context
from pydantic import BaseModel


class HeartBeat(BaseModel):
    type: str = "heartbeat"


async def heartbeat(context: Context, stop_event: asyncio.Event, interval: float = 30) -> None:
    """Send heartbeat messages every `interval` seconds until stop_event is set."""
    while not stop_event.is_set():
        await context.yield_async(HeartBeat())
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=interval)
        except TimeoutError:
            # timeout expired → loop again and send another heartbeat
            continue
