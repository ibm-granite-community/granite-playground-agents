# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import asyncio
from datetime import timedelta
from typing import TypeVar, cast

from acp_sdk import AsyncIterator, BaseModel
from acp_sdk.server import MemoryStore
from acp_sdk.server.store.store import Store, StoreModel, StoreView, T
from acp_sdk.server.store.utils import Stringable
from aiocache import Cache

U_co = TypeVar("U_co", bound=BaseModel, covariant=True)


class PrefixRouterMemoryStore(MemoryStore[T]):
    def __init__(self, *, limit: int = 1000, ttl: int | None = timedelta(hours=1)) -> None:  # type: ignore
        super().__init__(limit=limit, ttl=ttl)
        self.prefix_store_map: dict[str, Store] = {}

        # Use an async typed cache to avoid serialization and validation
        self._t_cache: Cache[str, str] = Cache(Cache.MEMORY, ttl=ttl)

        # Events watching on a specific key
        self._lock = asyncio.Lock()
        self._watchers: dict[str, set[asyncio.Event]] = {}
        self._batch_tasks: dict[str, asyncio.Task] = {}
        self._batch_delay = 0.03

    def map_prefix(self, prefix: str, store: Store) -> None:
        self.prefix_store_map[prefix] = store

    def as_store(self, model: type[U_co], prefix: Stringable = "") -> "Store[U_co]":
        if str(prefix) in self.prefix_store_map:
            return StoreView(model=model, store=self.prefix_store_map[str(prefix)], prefix=prefix)
        else:
            # Default to self memory store
            return StoreView(model=model, store=self, prefix=prefix)

    async def get(self, key: Stringable) -> T | None:
        value = await self._t_cache.get(str(key))
        return cast(T, StoreModel.model_validate_json(value)) if value else value

    async def set(self, key: Stringable, value: T | None) -> None:
        key_str = str(key)

        if value is None:
            await self._t_cache.pop(key_str, None)
        else:
            await self._t_cache.set(key_str, value.model_dump_json())

        async with self._lock:
            # schedule batch notification for this key if not already scheduled
            if key_str not in self._batch_tasks:
                self._batch_tasks[key_str] = asyncio.create_task(self._notify_watchers(key_str))

    async def _notify_watchers(self, key_str: str) -> None:
        await asyncio.sleep(self._batch_delay)  # debounce period

        async with self._lock:
            if key_str in self._watchers:
                for event in self._watchers[key_str]:
                    event.set()
                self._watchers[key_str].clear()
            self._batch_tasks.pop(key_str, None)

    async def watch(self, key: Stringable, *, ready: asyncio.Event | None = None) -> AsyncIterator[T | None]:
        """Watch on the store using a specific key"""
        key_str = str(key)
        if ready:
            ready.set()

        while True:
            event = asyncio.Event()

            async with self._lock:
                self._watchers.setdefault(key_str, set()).add(event)

            # Wait for key to be updated
            await event.wait()
            yield await self.get(key_str)
