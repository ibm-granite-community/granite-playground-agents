from abc import ABC, abstractmethod

from httpx import AsyncClient, Client


class AsyncScraper(ABC):
    @abstractmethod
    async def ascrape(self, link: str, client: AsyncClient) -> tuple:
        """Do scrape"""
        pass


class SyncScraper(ABC):
    @abstractmethod
    def scrape(self, link: str, client: Client) -> tuple:
        """Do scrape"""
        pass
