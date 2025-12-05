# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0


import pytest
from httpx import AsyncClient

from granite_core.search.scraping.types import ScrapedContent
from granite_core.search.scraping.wikipedia import WikipediaScraper
from granite_core.search.user_agent import UserAgent


@pytest.mark.asyncio
async def test_wikipedia_scraping() -> None:
    client = AsyncClient()
    client.headers.update({"User-Agent": UserAgent().user_agent})
    scraper = WikipediaScraper()
    content: ScrapedContent | None = await scraper.ascrape(
        link="https://en.wikipedia.org/wiki/IBM",
        client=client,
    )

    assert content is not None
    assert content.title == "IBM"
    assert len(content.content) > 0
