# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json

from granite_core.config import settings as core_settings
from granite_core.search.engines.engine import SearchEngine
from granite_core.search.engines.factory import SearchEngineFactory
from granite_core.search.scraping.runner import ScraperRunner
from granite_core.search.scraping.types import ScrapedSearchResult
from granite_core.search.types import SearchResult


async def scrape_urls(search_results: list[SearchResult]) -> list[ScrapedSearchResult]:
    scraper: ScraperRunner = ScraperRunner(
        session_id="",
        search_results=search_results,
        scraper_key="bs",
        max_scraped_content=core_settings.SEARCH_MAX_SCRAPED_CONTENT,
    )
    return await scraper.run()


def internet_search(query: str) -> str:
    """Run a web search."""
    engine: SearchEngine = SearchEngineFactory.create()
    results: list[SearchResult] = asyncio.run(main=engine.search(query=query, max_results=6))
    scraped_results: list[ScrapedSearchResult] = asyncio.run(main=scrape_urls(search_results=results))
    return json.dumps(
        obj=[{"title": s.title, "url": s.url, "content": s.raw_content} for s in scraped_results], indent=4
    )
