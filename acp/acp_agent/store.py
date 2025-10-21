# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0


# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import asyncio
from datetime import timedelta
from typing import TypeVar, cast

from acp_sdk import AsyncIterator, BaseModel
from acp_sdk.server import MemoryStore
from acp_sdk.server.store.store import Store, StoreModel, StoreView, T
from acp_sdk.server.store.utils import Stringable

from acp_agent.cache import AsyncLRUCache

U_co = TypeVar("U_co", bound=BaseModel, covariant=True)


class AsyncDebouncingMemoryStore(MemoryStore[T]):
    def __init__(
        self,
        *,
        limit: int = 100,
        ttl: int | None = timedelta(hours=1),  # type: ignore
        debounce: float = 0.2,
    ) -> None:
        super().__init__(limit=limit, ttl=ttl)
        self._t_cache: AsyncLRUCache[str, dict] = AsyncLRUCache(max_size=limit)
        self._lock = asyncio.Lock()
        self._watchers: dict[str, set[asyncio.Event]] = {}
        self._batch_tasks: dict[str, asyncio.Task] = {}
        self._batch_delay = debounce

    async def get(self, key: Stringable) -> T | None:
        value = await self._t_cache.get(str(key))
        return cast(T, StoreModel(**value)) if value else value

    async def set(self, key: Stringable, value: T | None) -> None:
        key_str = str(key)

        if value is None:
            await self._t_cache.delete(key_str)
        else:
            await self._t_cache.set(key_str, value.model_dump())

        # schedule batch notification for this key
        async with self._lock:
            if key_str not in self._batch_tasks:
                self._batch_tasks[key_str] = asyncio.create_task(self._notify_watchers(key_str))

    async def _notify_watchers(self, key_str: str) -> None:
        await asyncio.sleep(self._batch_delay)  # debounce period

        async with self._lock:
            if key_str in self._watchers:
                for event in self._watchers[key_str]:
                    event.set()
                self._watchers[key_str].clear()
                self._watchers.pop(key_str, None)
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


class PrefixRouterMemoryStore(AsyncDebouncingMemoryStore[T]):
    def __init__(
        self,
        *,
        limit: int = 100,
        ttl: int | None = timedelta(hours=1),  # type: ignore
        debounce: float = 0.2,
    ) -> None:
        super().__init__(limit=limit, ttl=ttl, debounce=debounce)
        self.prefix_store_map: dict[str, Store] = {}

    def map_prefix(self, prefix: str, store: Store) -> None:
        self.prefix_store_map[prefix] = store

    def as_store(self, model: type[U_co], prefix: Stringable = "") -> "Store[U_co]":
        if str(prefix) in self.prefix_store_map:
            return StoreView(model=model, store=self.prefix_store_map[str(prefix)], prefix=prefix)
        else:
            # Default to self memory store
            return StoreView(model=model, store=self, prefix=prefix)
