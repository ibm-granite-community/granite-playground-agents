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

from httpx import AsyncClient, Client

from granite_core.config import settings
from granite_core.emitter import EventEmitter
from granite_core.events import TrajectoryEvent
from granite_core.logging import get_logger_with_prefix
from granite_core.search.scraping import BeautifulSoupScraper
from granite_core.search.scraping.arxiv import ArxivScraper
from granite_core.search.scraping.pymupdf import PyMuPDFScraper
from granite_core.search.scraping.scraper import AsyncScraper, SyncScraper
from granite_core.search.types import ScrapedContent, SearchResult
from granite_core.work import task_pool


class ContentExtractor(EventEmitter):
    """
    Scraper class to extract the content from the links
    """

    def __init__(
        self,
        search_results: list[SearchResult],
        user_agent: str,
        scraper_key: str,
        session_id: str,
        max_scraped_content: int = 10,
    ) -> None:
        """
        Initialize the Scraper class.
        Args:
            urls:
        """
        super().__init__()
        self.search_results = search_results
        self.async_client = AsyncClient()
        self.client = Client()
        self.async_client.headers.update({"User-Agent": user_agent})
        self.client.headers.update({"User-Agent": user_agent})
        self._counter_lock = asyncio.Lock()
        self._content_count: int = 0
        self._max_scraped_content = max_scraped_content
        self.scraper_key = scraper_key
        self.logger = get_logger_with_prefix(__name__, tool_name="ContentExtractor", session_id=session_id)

    async def run(self) -> list[ScrapedContent]:
        """
        Extracts the content from the links
        """
        contents = await asyncio.gather(*(self.extract_data_from_url(s) for s in self.search_results))
        res = [content for content in contents if content is not None]
        return res

    async def extract_data_from_url(self, search_result: SearchResult) -> ScrapedContent | None:
        """
        Extracts the data from the link with logging
        """
        if self._content_count >= self._max_scraped_content:
            self.logger.info("Max scraped content exceeded!")
            return None

        try:
            link = search_result.url
            scraper_cls: type[AsyncScraper] | type[SyncScraper] = self.get_scraper(link)
            scraper = scraper_cls()

            # Get scraper name
            scraper_name = scraper.__class__.__name__
            self.logger.info(f"=== Using {scraper_name} ===")

            # Get content
            async with task_pool.throttle():
                if isinstance(scraper, AsyncScraper):
                    content, image_urls, title = await asyncio.wait_for(
                        cast(AsyncScraper, scraper).ascrape(link=link, client=self.async_client),
                        timeout=settings.SCRAPER_TIMEOUT,
                    )
                else:
                    loop = asyncio.get_running_loop()
                    sync_scraper = cast(SyncScraper, scraper)
                    (
                        content,
                        image_urls,
                        title,
                    ) = await asyncio.wait_for(
                        loop.run_in_executor(task_pool.executor, lambda: sync_scraper.scrape(link, self.client)),
                        timeout=settings.SCRAPER_TIMEOUT,
                    )

            if len(content) < 200:
                self.logger.warning(f"Content too short or empty for {link}")
                return None

            # Log results
            self.logger.info(f"Title: {title}")
            self.logger.info(f"Content length: {len(content) if content else 0} characters")
            # self.logger.info(f"Number of images: {len(image_urls)}")
            self.logger.info(f"URL: {link}")
            self.logger.info("=" * 50)

            await self._emit(TrajectoryEvent(title="Added source", content=link))

            async with self._counter_lock:
                self._content_count += 1

            return ScrapedContent(
                search_result=search_result,
                url=link,
                raw_content=content[: settings.SCRAPER_MAX_CONTENT_LENGTH],
                # image_urls=image_urls,
                title=title,
            )

        except TimeoutError as e:
            self.logger.error(f"Timed out scraping {link}: {e!s}")
            return None

        except Exception as e:
            self.logger.error(f"Error processing {link}: {e!s}")
            return None

    def get_scraper(
        self,
        link: str,
    ) -> type[AsyncScraper] | type[SyncScraper]:
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

        scraper_classes: dict[str, type[AsyncScraper] | type[SyncScraper]] = {
            "pdf": PyMuPDFScraper,
            "bs": BeautifulSoupScraper,
            "arxiv": ArxivScraper,
        }

        scraper_key = None

        if link.endswith(".pdf"):
            scraper_key = "pdf"
        elif "arxiv.org" in link:
            scraper_key = "arxiv"
        else:
            scraper_key = self.scraper_key

        scraper_class = scraper_classes.get(scraper_key)

        if scraper_class is None:
            raise Exception("Scraper not found.")

        return scraper_class
