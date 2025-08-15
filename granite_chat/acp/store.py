# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import asyncio
from collections.abc import AsyncIterator
from datetime import timedelta
from typing import Generic, TypeVar

from acp_sdk import BaseModel
from acp_sdk.server import MemoryStore, RedisStore
from acp_sdk.server.store.store import Store, StoreView, T
from acp_sdk.server.store.utils import Stringable
from redis.asyncio import Redis

from granite_chat.config import settings

U_co = TypeVar("U_co", bound=BaseModel, covariant=True)

SESSION_PREFIX = "session_"


class PrefixRouterStore(Store[T], Generic[T]):
    def __init__(self) -> None:
        super().__init__()

        self.prefix_store_map: dict[str, Store] = {}

        if settings.KEY_STORE_PROVIDER == "redis":
            redis = Redis().from_url(settings.REDIS_CLIENT_URL)
            """Sessions are stored in persistent store, everything else to memory"""
            self.prefix_store_map[SESSION_PREFIX] = RedisStore(redis=redis)

        self._memory_store: MemoryStore = MemoryStore(limit=1000, ttl=timedelta(hours=1))  # type: ignore

    def as_store(self, model: type[U_co], prefix: Stringable = "") -> "Store[U_co]":
        if str(prefix) in self.prefix_store_map:
            return StoreView(model=model, store=self.prefix_store_map[str(prefix)], prefix=prefix)
        else:
            return StoreView(model=model, store=self._memory_store, prefix=prefix)

    async def get(self, key: Stringable) -> T | None:
        # Handled by StoreView
        pass

    async def set(self, key: Stringable, value: T | None) -> None:
        # Handled by StoreView
        pass

    async def watch(self, key: Stringable, *, ready: asyncio.Event | None = None) -> AsyncIterator[T | None]:
        # Handled by StoreView
        if False:  # never runs, but type checker sees yield
            yield 0
        return
