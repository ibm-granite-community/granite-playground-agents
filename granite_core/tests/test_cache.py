# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0


import pytest

from granite_core.cache import AsyncLRUCache


@pytest.mark.asyncio
async def test_async_cache() -> None:
    cache: AsyncLRUCache[str, int] = AsyncLRUCache[str, int](max_size=3)

    await cache.set("a", 1)
    await cache.set("b", 2)
    await cache.set("c", 3)
    await cache.set("d", 4)

    # a was kicked due to maxsize
    assert not await cache.exists("a")

    await cache.delete("b")

    assert not await cache.exists("b")

    # Moves c to top of cache
    assert await cache.get("c") == 3

    # insert e, no change
    await cache.set("e", 5)

    assert await cache.exists("d")

    # insert f, ejects d
    await cache.set("f", 6)

    assert not await cache.exists("d")
