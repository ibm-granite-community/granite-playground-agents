# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0


from abc import ABC, abstractmethod

from httpx import AsyncClient

from granite_core.search.scraping.types import ScrapedContent


class AsyncScraper(ABC):
    @abstractmethod
    async def ascrape(self, link: str, client: AsyncClient) -> ScrapedContent | None:
        """Do scrape"""
        pass
