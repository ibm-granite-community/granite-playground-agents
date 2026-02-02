# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0


# Portions of this file are derived from the Apache 2.0 licensed project "gpt-researcher"
# Original source: https://github.com/assafelovic/gpt-researcher
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Changes made:

import asyncio
from typing import cast

from httpx import AsyncClient, Timeout

from granite_core.config import settings
from granite_core.emitter import EventEmitter
from granite_core.events import TrajectoryEvent
from granite_core.logging import get_logger_with_prefix
from granite_core.search.scraping.arxiv import ArxivScraper
from granite_core.search.scraping.base import AsyncScraper
from granite_core.search.scraping.beautiful_soup import BeautifulSoupScraper
from granite_core.search.scraping.docling import DoclingPDFScraper
from granite_core.search.scraping.types import ScrapedContent
from granite_core.search.scraping.wikipedia import WikipediaScraper
from granite_core.search.user_agent import UserAgent
from granite_core.work import task_pool


class ScraperRunner(EventEmitter):
    """
    Scraper class to extract the content from the links
    """

    def __init__(
        self,
        urls: list[str],
        scraper_key: str = "bs",
        session_id: str = "",
        max_scraped_content: int = 10,
    ) -> None:
        """
        Initialize the Scraper class.
        Args:
            urls:
        """
        super().__init__()
        self.urls = urls
        self.async_client = AsyncClient(timeout=Timeout(connect=5, read=8, write=5, pool=5))
        self.async_client.headers.update(headers={"User-Agent": UserAgent().user_agent})
        self._counter_lock = asyncio.Lock()
        self._content_count: int = 0
        self._max_scraped_content = max_scraped_content
        self.scraper_key = scraper_key
        self.logger = get_logger_with_prefix(__name__, tool_name=__name__, session_id=session_id)

    async def run(self) -> list[ScrapedContent]:
        """
        Extracts the content from the links
        """
        contents = await asyncio.gather(*(self.scrape_data_from_url(url) for url in self.urls))
        res = [content for content in contents if content is not None]
        return res

    async def scrape_data_from_url(self, url: str) -> ScrapedContent | None:
        """
        Extracts the data from the link with logging
        """
        if self._content_count >= self._max_scraped_content:
            self.logger.info("Max scraped content exceeded!")
            return None

        try:
            scraper_cls: type[AsyncScraper] = self.get_scraper(url)
            scraper = scraper_cls()

            # Get scraper name
            scraper_name = scraper.__class__.__name__
            self.logger.info(f"=== Using {scraper_name} ===")

            # Get content
            async with task_pool.throttle():
                scraped_content: ScrapedContent | None = await asyncio.wait_for(
                    fut=cast(AsyncScraper, scraper).ascrape(link=url, client=self.async_client),
                    timeout=settings.SCRAPER_TIMEOUT,
                )

            if scraped_content is None:
                self.logger.warning(f"No scraped result for {url}")
                return None

            if scraped_content.content is None or len(scraped_content.content) < 200:
                self.logger.warning(f"Content too short or empty for {url}")
                return None

            # Log results
            self.logger.info(f"Title: {scraped_content.title}")
            self.logger.info(f"Content length: {len(scraped_content.content)} characters")
            # self.logger.info(f"Number of images: {len(image_urls)}")
            self.logger.info(f"URL: {url}")
            self.logger.info("=" * 50)

            await self._emit(TrajectoryEvent(title="Added source", content=url))

            async with self._counter_lock:
                self._content_count += 1

            return scraped_content

        except TimeoutError as e:
            self.logger.error(f"Timed out scraping {url}: {e!s}")
            return None

        except Exception as e:
            self.logger.error(f"Error processing {url}: {e!s}")
            return None

    def get_scraper(
        self,
        link: str,
    ) -> type[AsyncScraper]:
        """
        The function `get_scraper` determines the appropriate scraper class based on the provided link
        or a default scraper if none matches.

        Args:
          link: The `get_scraper` method takes a `link` parameter which is a URL link to a webpage or a
        PDF file. Based on the type of content the link points to, the method determines the appropriate
        scraper class to use for extracting data from that content.

        Returns:
          The `get_scraper` method returns the scraper class based on the provided link. The method
        checks the link to determine the appropriate scraper class to use based on predefined mappings
        in the `SCRAPER_CLASSES` dictionary. If the link ends with ".pdf", it selects the
        `PyMuPDFScraper` class. If the link contains "arxiv.org", it selects the `ArxivScraper
        """

        scraper_classes: dict[str, type[AsyncScraper]] = {
            "pdf": DoclingPDFScraper,
            "bs": BeautifulSoupScraper,
            "arxiv": ArxivScraper,
            "wikipedia": WikipediaScraper,
        }

        scraper_key = None

        if link.endswith(".pdf"):
            scraper_key = "pdf"
        elif "arxiv.org" in link:
            scraper_key = "arxiv"
        elif "en.wikipedia.org/wiki/" in link:
            scraper_key = "wikipedia"
        else:
            scraper_key = self.scraper_key

        scraper_class = scraper_classes.get(scraper_key)

        if scraper_class is None:
            raise Exception("Scraper not found.")

        return scraper_class
