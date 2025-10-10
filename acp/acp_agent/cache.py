import asyncio
from collections import OrderedDict
from typing import Generic, TypeVar

K = TypeVar("K", bound=str)  # Key type
V = TypeVar("V")  # Value type


class AsyncLRUCache(Generic[K, V]):
    def __init__(self, max_size: int) -> None:
        self._cache: OrderedDict[K, V] = OrderedDict()
        self._max_size = max_size
        self._lock = asyncio.Lock()

    async def exists(self, key: K) -> bool:
        async with self._lock:
            return key in self._cache

    async def get(self, key: K) -> V | None:
        async with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)  # mark as recently used
                return self._cache[key]
            return None

    async def set(self, key: K, value: V) -> None:
        async with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            self._cache[key] = value
            if len(self._cache) > self._max_size:
                self._cache.popitem(last=False)  # remove least recently used

    async def delete(self, key: K) -> None:
        async with self._lock:
            self._cache.pop(key, None)
