# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0

import json

from mcp.server.fastmcp import FastMCP

from granite_core.search.engines.engine import SearchEngine
from granite_core.search.engines.factory import SearchEngineFactory
from granite_core.search.scraping.runner import ScraperRunner
from granite_core.search.scraping.types import ScrapedContent
from granite_core.search.types import SearchResult
from granite_core.config import settings

mcp = FastMCP("internet_search", port=8001)

MAX_SEARCH_RESULTS = 10


async def scrape_urls(urls: list[str]) -> list[ScrapedContent]:
    scraper = ScraperRunner(
        urls=urls,
        scraper_key="bs",
        max_scraped_content=settings.SEARCH_MAX_SCRAPED_CONTENT,
    )
    return await scraper.run()


@mcp.tool(
    name="internet_search",
    description="Searches the internet.",
)
async def internet_search(query: str) -> str:
    engine: SearchEngine = SearchEngineFactory.create()
    results: list[SearchResult] = await engine.search(
        query=query, max_results=MAX_SEARCH_RESULTS
    )
    contents: list[ScrapedContent] = await scrape_urls([r.url for r in results])
    if not contents:
        return "No internet search results found."
    return json.dumps(
        obj=[{"title": c.title, "url": c.url, "content": c.content} for c in contents],
        indent=4,
    )


def main():
    # mcp.run(transport="stdio")
    mcp.run(transport="streamable-http")


if __name__ == "__main__":
    main()
