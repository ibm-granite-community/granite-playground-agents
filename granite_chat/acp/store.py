# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from datetime import timedelta
from typing import TypeVar

from acp_sdk import BaseModel
from acp_sdk.server import MemoryStore
from acp_sdk.server.store.store import Store, StoreView, T
from acp_sdk.server.store.utils import Stringable

U_co = TypeVar("U_co", bound=BaseModel, covariant=True)


class PrefixRouterMemoryStore(MemoryStore[T]):
    def __init__(self, *, limit: int = 1000, ttl: int | None = timedelta(hours=1)) -> None:  # type: ignore
        super().__init__(limit=limit, ttl=ttl)
        self.prefix_store_map: dict[str, Store] = {}

    def map_prefix(self, prefix: str, store: Store) -> None:
        self.prefix_store_map[prefix] = store

    def as_store(self, model: type[U_co], prefix: Stringable = "") -> "Store[U_co]":
        if str(prefix) in self.prefix_store_map:
            return StoreView(model=model, store=self.prefix_store_map[str(prefix)], prefix=prefix)
        else:
            # Default to self memory store
            return StoreView(model=model, store=self, prefix=prefix)
