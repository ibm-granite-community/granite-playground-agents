# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0

import time
from logging import Logger
from urllib.parse import ParseResult, urlparse
from urllib.robotparser import RobotFileParser

from httpx import AsyncClient
from httpx._models import Response
from pydantic import BaseModel, ConfigDict

from granite_core.cache import AsyncLRUCache
from granite_core.logging import get_logger


class MutableRobotFileParser(RobotFileParser):
    # Setter for disallow_all
    def set_disallow_all(self, value: bool) -> None:
        self.disallow_all = value

    # Setter for allow_all
    def set_allow_all(self, value: bool) -> None:
        self.allow_all = value


class CachedRobotParser(BaseModel):
    parser: MutableRobotFileParser
    timestamp: float
    model_config = ConfigDict(arbitrary_types_allowed=True)


logger: Logger = get_logger(logger_name=__name__)
_robot_cache: AsyncLRUCache[str, CachedRobotParser] = AsyncLRUCache[str, CachedRobotParser](max_size=500)
CACHE_TTL = 604800  # 7 days in seconds: 7 * 24 * 60 * 60


async def get_robots_parser(client: AsyncClient, robots_url: str, user_agent: str = "*") -> MutableRobotFileParser:
    cached_parser: CachedRobotParser | None = await _robot_cache.get(key=robots_url)

    # If nothing in cache, or what we have in cache is stale then we need to fetch it
    if not cached_parser or (cached_parser and (time.time() - cached_parser.timestamp > CACHE_TTL)):
        logger.info(msg=f"Need to load {robots_url}, it is not available or stale.")
        rp: MutableRobotFileParser = MutableRobotFileParser()

        try:
            async with AsyncClient(timeout=5.0, headers={"User-Agent": user_agent}) as client:
                response: Response = await client.get(url=robots_url, follow_redirects=True)

                if response.status_code == 200:
                    rp.parse(lines=response.text.splitlines())
                elif response.status_code in [401, 403]:
                    # If forbidden, robots.txt logic usually says 'disallow all'
                    logger.info(msg=f"Robots.txt for {robots_url} is forbidden! Assuming disallowed.")
                    rp.set_disallow_all(value=True)
                else:
                    # If not found, robots.txt logic usually says 'allow all'
                    logger.info(msg=f"Robots.txt for {robots_url} is unavailable! Assuming allowed.")
                    rp.set_allow_all(value=True)

        except Exception:
            # If robots.txt is unreachable, treat as allowed
            logger.info(msg=f"Robots.txt for {robots_url} is unavailable! Assuming allowed.")
            rp.set_allow_all(value=True)

        cached_parser = CachedRobotParser(parser=rp, timestamp=time.time())
        await _robot_cache.set(key=robots_url, value=cached_parser)

    return cached_parser.parser


async def can_fetch(client: AsyncClient, url: str, user_agent: str = "*") -> bool:
    parsed: ParseResult = urlparse(url)
    robots_url: str = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
    parser: MutableRobotFileParser = await get_robots_parser(
        robots_url=robots_url, client=client, user_agent=user_agent
    )
    allow_fetch: bool = parser.can_fetch(useragent=user_agent, url=url)
    logger.info(msg=f"Robots.txt for {robots_url} {'allows' if allow_fetch else 'disallows'} fetching {url}")
    return allow_fetch
