# Portions of this file are derived from the Apache 2.0 licensed project "gpt-researcher"
# Original source: https://github.com/assafelovic/gpt-researcher/blob/master/gpt_researcher/retrievers/google/google.py
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Changes made:

import json
import logging
import os

import httpx

from granite_chat.search.engines.engine import SearchEngine

logger = logging.getLogger(__name__)


class GoogleSearch(SearchEngine):
    """
    Google API Retriever
    """

    def __init__(self) -> None:
        """
        Initializes the GoogleSearch object
        Args:
            query:
        """
        self.api_key = self.get_api_key()
        self.cx_key = self.get_cx_key()

    def get_api_key(self) -> str:
        """
        Gets the Google API key
        Returns:

        """
        # Get the API key
        try:
            api_key = os.environ["GOOGLE_API_KEY"]
        except Exception:
            raise Exception(  # noqa: B904
                "Google API key not found. Please set the GOOGLE_API_KEY environment variable. "
                "You can get a key at https://developers.google.com/custom-search/v1/overview"
            )
        return api_key

    def get_cx_key(self) -> str:
        """
        Gets the Google CX key
        Returns:

        """
        # Get the API key
        try:
            api_key = os.environ["GOOGLE_CX_KEY"]
        except Exception:
            raise Exception(  # noqa: B904
                "Google CX key not found. Please set the GOOGLE_CX_KEY environment variable. "
                "You can get a key at https://developers.google.com/custom-search/v1/overview"
            )
        return api_key

    async def search(self, query: str, domains: list[str] | None = None, max_results: int = 7) -> list[dict[str, str]]:
        """
        Searches the query using Google Custom Search API, optionally restricting to specific domains
        Returns:
            list: List of search results with title, href and body
        """
        # Build query with domain restrictions if specified
        if domains and len(domains) > 0:
            domain_query = " OR ".join([f"site:{domain}" for domain in domains])
            query = f"({domain_query}) {query}"

        url = f"https://www.googleapis.com/customsearch/v1?key={self.api_key}&cx={self.cx_key}&q={query}&start=1"

        async with httpx.AsyncClient() as client:
            resp = await client.get(url)

            if resp.status_code < 200 or resp.status_code >= 300:
                logger.warning("Google search: unexpected response status: ", resp.status_code)

            if resp is None:
                return [{}]
            try:
                search_results = json.loads(resp.text)
            except Exception:
                return [{}]
            if search_results is None:
                return [{}]

            results = search_results.get("items", [])
            search_results = []

            # Normalizing results to match the format of the other search APIs
            for result in results:
                # skip youtube results
                if "youtube.com" in result["link"]:
                    continue
                try:
                    search_result = {
                        "title": result["title"],
                        "href": result["link"],
                        "body": result["snippet"],
                    }
                except Exception:
                    continue
                search_results.append(search_result)

            return search_results[:max_results]
