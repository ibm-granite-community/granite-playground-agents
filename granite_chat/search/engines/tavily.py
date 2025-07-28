# Portions of this file are derived from the Apache 2.0 licensed project "gpt-researcher"
# Original source: https://github.com/assafelovic/gpt-researcher/blob/master/gpt_researcher/retrievers/tavily/tavily.py
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Changes made:

from tavily import AsyncTavilyClient

from granite_chat.config import settings
from granite_chat.search.engines.engine import SearchEngine
from granite_chat.search.types import SearchResult
from granite_chat.work import task_pool


class TavilySearch(SearchEngine):
    """
    Tavily API Retriever
    """

    def __init__(self) -> None:
        """
        Initializes the TavilySearch object
        Args:
            query:
        """
        self.api_key = settings.TAVILY_API_KEY
        self.tavily_client = AsyncTavilyClient(self.api_key)

    async def search(self, query: str, domains: list[str] | None = None, max_results: int = 7) -> list[SearchResult]:
        """
        Searches the query using Tavily Search API, optionally restricting to specific domains
        Returns:
            list: List of search results with title, href and body
        """
        async with task_pool.throttle():
            results = await self.tavily_client.search(query=query, max_results=max_results, domains=domains)
            search_results = []

            # Normalizing results to match the format of the other search APIs
            for result in results["results"]:
                # skip youtube results
                if "youtube.com" in result["url"]:
                    continue
                try:
                    search_result = SearchResult(
                        title=result["title"],
                        href=result["link"],
                        body=result["snippet"],
                    )
                except Exception:
                    continue
                search_results.append(search_result)

            return search_results
