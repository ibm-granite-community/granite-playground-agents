# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0


from abc import ABC, abstractmethod

from httpx import AsyncClient

from granite_core.config import settings
from granite_core.logging import get_logger
from granite_core.search.robots import can_fetch
from granite_core.search.scraping.types import ScrapedContent
from granite_core.search.user_agent import UserAgent

logger = get_logger(__name__)


class AsyncScraper(ABC):
    @abstractmethod
    async def ascrape(self, link: str, client: AsyncClient) -> ScrapedContent | None:
        """Do scrape"""
        pass

    async def can_scrape(self, client: AsyncClient, link: str) -> bool:
        if settings.CHECK_ROBOTS_TXT:
            allowed = await can_fetch(client=client, url=link, user_agent=UserAgent().user_agent)

            if not allowed:
                logger.info(f"Not allowed to scrape {link} due to robots.txt")
                return False

        return True
