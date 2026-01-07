# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json

from mcp.server.fastmcp import FastMCP

from granite_core.search.engines.factory import SearchEngineFactory
from granite_core.search.scraping.runner import ScraperRunner
from granite_core.search.scraping.types import ScrapedContent
from granite_core.search.types import SearchResult
from granite_core.config import settings

mcp = FastMCP("web")


async def search_query(query: str, max_results) -> list[SearchResult]:
    engine = SearchEngineFactory.create()
    results = await engine.search(query=query, max_results=max_results)
    return results


@mcp.tool(
    name="search",
    description="Searches the web and returns a list of search results",
)
async def search(queries: list[str]) -> str:
    results: list[list[SearchResult]] = await asyncio.gather(
        *(search_query(q, settings.SEARCH_MAX_SEARCH_RESULTS_PER_STEP) for q in queries)
    )
    return json.dumps([item.model_dump() for sublist in results for item in sublist])


async def scrape_urls(urls: list[str]) -> list[ScrapedContent]:
    scraper = ScraperRunner(
        urls=urls,
        scraper_key="bs",
        max_scraped_content=settings.SEARCH_MAX_SCRAPED_CONTENT,
    )
    return await scraper.run()


@mcp.tool(
    name="scrape",
    description="Fetch and extract readable text content from a set of web URLs.",
)
async def scrape(urls: list[str]) -> str:
    scraped_contents: list[ScrapedContent] = await scrape_urls(urls=urls)
    return json.dumps([item.model_dump() for item in scraped_contents])


def main():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
