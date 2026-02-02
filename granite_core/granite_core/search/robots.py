# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0

from logging import Logger
from urllib.parse import ParseResult, urlparse
from urllib.robotparser import RobotFileParser

from httpx import AsyncClient
from httpx._models import Response

from granite_core.cache import AsyncLRUCache
from granite_core.logging import get_logger

logger: Logger = get_logger(logger_name=__name__)
_robot_cache: AsyncLRUCache[str, RobotFileParser] = AsyncLRUCache[str, RobotFileParser](max_size=1000)


async def can_fetch(client: AsyncClient, url: str, user_agent: str = "*") -> bool:
    parsed: ParseResult = urlparse(url)
    robots_url: str = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
    if not await _robot_cache.exists(key=robots_url):
        rp: RobotFileParser = RobotFileParser()

        try:
            async with AsyncClient(timeout=5.0) as client:
                response: Response = await client.get(url=robots_url, follow_redirects=True)
                rp.parse(lines=response.text.splitlines())
        except Exception:
            logger.info(msg=f"Robots.txt for {robots_url} is unavailable! Assuming allowed.")
            return True  # If robots.txt is unreachable, treat as allowed

        await _robot_cache.set(key=robots_url, value=rp)

    cached_rp: RobotFileParser | None = await _robot_cache.get(key=robots_url)
    allow_fetch: bool = cached_rp.can_fetch(useragent=user_agent, url=url) if cached_rp else True
    logger.info(msg=f"Robots.txt for {robots_url} {'allows' if allow_fetch else 'disallows'} fetching {url}")
    return allow_fetch
