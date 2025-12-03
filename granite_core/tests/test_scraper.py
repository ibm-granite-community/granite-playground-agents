# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0


import pytest

from granite_core.search.scraping import scrape_search_results
from granite_core.search.types import SearchResult


@pytest.mark.asyncio
async def test_scraper() -> None:
    """Test scraping infra"""
    search_result = SearchResult(title="IBM", body="", href="https://www.ibm.com/about")

    results, _ = await scrape_search_results(
        search_results=[search_result], scraper_key="bs", session_id="", max_scraped_content=1
    )

    assert len(results) == 1
