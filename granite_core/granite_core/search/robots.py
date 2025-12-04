# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0

import asyncio
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

from granite_core.cache import AsyncLRUCache
from granite_core.logging import get_logger_with_prefix
from granite_core.search.types import SearchResult
from granite_core.search.user_agent import UserAgent

_robot_cache: AsyncLRUCache[str, RobotFileParser] = AsyncLRUCache[str, RobotFileParser](max_size=1000)


async def can_fetch(url: str, user_agent: str = "*") -> bool:
    parsed = urlparse(url)
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
    if not await _robot_cache.exists(robots_url):
        rp = RobotFileParser()
        try:
            rp.set_url(robots_url)
            await asyncio.to_thread(rp.read)
        except Exception:
            # If robots.txt is unreachable, treat as allowed (common crawler behavior)
            return True
        await _robot_cache.set(robots_url, rp)

    cached_rp = await _robot_cache.get(robots_url)
    allow_fetch = cached_rp.can_fetch(useragent=user_agent, url=url) if cached_rp else True
    return allow_fetch


class RobotsTxtFilter:
    def __init__(self, session_id: str) -> None:
        self.logger = get_logger_with_prefix(__name__, tool_name="RobotsTxtFilter", session_id=session_id)

    async def filter(self, results: list[SearchResult]) -> list[SearchResult]:
        filtered_results = await asyncio.gather(*(self._filter_search_result(result) for result in results))
        return [r for r in filtered_results if r is not None]

    async def _filter_search_result(self, result: SearchResult) -> SearchResult | None:
        allowed = await can_fetch(user_agent=UserAgent().user_agent(), url=result.href)
        self.logger.info(f"Allow scrape {result.href} = {allowed}")
        return result if allowed else None
