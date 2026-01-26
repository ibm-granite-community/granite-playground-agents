# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0

"""Internet search MCP service."""

from argparse import ArgumentParser, Namespace


import json

from granite_core.search.engines.engine import SearchEngine
from granite_core.search.engines.factory import SearchEngineFactory
from granite_core.search.scraping.runner import ScraperRunner
from granite_core.search.scraping.types import ScrapedContent
from granite_core.search.types import SearchResult

from granite_core_mcp.base import MCPService, TransportType


class InternetSearchService(MCPService):
    """MCP service for internet search functionality.

    Provides a tool to search the internet and scrape content from results.
    """

    def __init__(
        self,
        port: int = 8001,
        transport: TransportType = "streamable-http",
        max_search_results: int = 10,
        max_scraped: int = 10,
        max_scraped_content_length: int = 10000,
    ) -> None:
        """Initialize the Internet Search service.

        Args:
            port: Port number for HTTP transport (default: 8001)
            transport: Transport type - "stdio", "sse", or "streamable-http" (default: "streamable-http")
            max_search_results: Maximum number of search results to return (default: 10)
        """
        self.max_search_results = max_search_results
        self.max_scraped = max_scraped
        self.max_scraped_content_length = max_scraped_content_length
        super().__init__(name="internet_search", port=port, transport=transport)

    def _register_tools(self) -> None:
        """Register the internet_search tool."""

        @self.mcp.tool(
            name="internet_search",
            description="Searches the internet.",
        )
        async def internet_search(query: str) -> str:
            """Search the internet for a query.

            Args:
                query: The search query string

            Returns:
                JSON string containing search results with title, url, and content
            """
            engine: SearchEngine = SearchEngineFactory.create()
            results: list[SearchResult] = await engine.search(
                query=query, max_results=self.max_search_results
            )

            scraper: ScraperRunner = ScraperRunner(
                urls=[r.url for r in results],
                scraper_key="bs",
                max_scraped_content=self.max_scraped,
            )

            contents: list[ScrapedContent] = await scraper.run()

            for sc in contents:
                if len(sc.content) > self.max_scraped_content_length:
                    sc.content = sc.content[: self.max_scraped_content_length] + "..."

            # If nothing was scrape-able
            if not contents:
                return "No internet search results found."

            return json.dumps(
                obj=[
                    {"title": c.title, "url": c.url, "content": c.content}
                    for c in contents
                ],
                indent=4,
            )


def main() -> None:
    """Run the Internet Search MCP service."""
    import argparse

    parser: ArgumentParser = argparse.ArgumentParser(
        description="Internet Search MCP Service"
    )
    parser.add_argument(
        "--port", type=int, default=8001, help="Port number (default: 8001)"
    )
    parser.add_argument(
        "--transport",
        type=str,
        default="streamable-http",
        choices=["stdio", "sse", "streamable-http"],
        help="Transport type (default: streamable-http)",
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=10,
        dest="max_results",
        help="Maximum number of search results (default: 10)",
    )

    args: Namespace = parser.parse_args()

    service: InternetSearchService = InternetSearchService(
        port=args.port,
        transport=args.transport,  # type: ignore[arg-type]
        max_search_results=args.max_results,
    )
    service.run()


if __name__ == "__main__":
    main()
