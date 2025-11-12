# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0


import asyncio

import pytest
from httpx import AsyncClient

from granite_core.search.scraping.arxiv import ArxivScraper


@pytest.fixture
def client() -> AsyncClient:
    return AsyncClient()


@pytest.fixture
def scraper() -> ArxivScraper:
    return ArxivScraper()


@pytest.mark.asyncio
async def test_arxix_scrape(scraper: ArxivScraper, client: AsyncClient) -> None:
    links = [
        "https://arxiv.org/abs/2402.05749v2",
        "https://arxiv.org/abs/2303.12712",
        "https://arxiv.org/abs/1706.03762",
    ]

    tasks = [scraper.ascrape(link, client) for link in links]
    results = await asyncio.gather(*tasks)

    assert len(results) == len(links)

    for r in results:
        assert r is not None
        assert r.content is not None
        assert r.title is not None
        assert len(r.title) > 0
        assert len(r.content) > 0
