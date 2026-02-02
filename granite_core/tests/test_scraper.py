# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0


import pytest

from granite_core.search.scraping import scrape_search_results
from granite_core.search.scraping.types import ScrapedSearchResult
from granite_core.search.types import SearchResult


@pytest.mark.asyncio
async def test_scraper() -> None:
    """Test scraping infra"""
    search_result: SearchResult = SearchResult(title="IBM", snippet="", url="https://www.ibm.com/about")

    results: list[ScrapedSearchResult] = await scrape_search_results(
        search_results=[search_result], scraper_key="bs", session_id="", max_scraped_content=1
    )

    assert len(results) == 1

    search_result = SearchResult(
        title="Bestbuy",
        snippet="",
        url="https://www.bestbuy.com/product/panasonic-streaming-4k-ultra-hd-hi-res-audio-with-dolby-vision-7-1-channel-dvd-cd-3d-wi-fi-built-in-blu-ray-player-dp-ub820-k-black/J3Z7HSJPYW",
    )

    results = await scrape_search_results(
        search_results=[search_result], scraper_key="bs", session_id="", max_scraped_content=1
    )

    assert len(results) == 0
