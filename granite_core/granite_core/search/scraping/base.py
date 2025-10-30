# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0


from abc import ABC, abstractmethod

from httpx import AsyncClient, Client

from granite_core.search.scraping.types import ScrapedContent


class AsyncScraper(ABC):
    @abstractmethod
    async def ascrape(self, link: str, client: AsyncClient) -> ScrapedContent | None:
        """Do scrape"""
        pass


class SyncScraper(ABC):
    @abstractmethod
    def scrape(self, link: str, client: Client) -> ScrapedContent | None:
        """Do scrape"""
        pass
