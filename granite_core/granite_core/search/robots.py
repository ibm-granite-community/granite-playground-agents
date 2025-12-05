# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0

import asyncio
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

from granite_core.cache import AsyncLRUCache

_robot_cache: AsyncLRUCache[str, RobotFileParser] = AsyncLRUCache[str, RobotFileParser](max_size=1000)


async def can_fetch(url: str, user_agent: str = "*") -> bool:
    parsed = urlparse(url)
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
    if not await _robot_cache.exists(robots_url):
        rp = RobotFileParser()
        try:
            rp.set_url(robots_url)
            await asyncio.wait_for(asyncio.to_thread(rp.read), timeout=5)
        except Exception:
            # If robots.txt is unreachable, treat as allowed (common crawler behavior)
            return True
        await _robot_cache.set(robots_url, rp)

    cached_rp = await _robot_cache.get(robots_url)
    allow_fetch = cached_rp.can_fetch(useragent=user_agent, url=url) if cached_rp else True
    return allow_fetch
