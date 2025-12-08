# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0


# Portions of this file are derived from the Apache 2.0 licensed project "gpt-researcher"
# Original source: https://github.com/assafelovic/gpt-researcher
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Changes made:


from colorama import Fore, Style

from granite_core.emitter import EventEmitter
from granite_core.logging import get_logger
from granite_core.search.scraping.runner import ScraperRunner
from granite_core.search.scraping.types import ImageUrl, ScrapedSearchResult
from granite_core.search.types import SearchResult

logger = get_logger(__name__)


async def scrape_search_results(
    search_results: list[SearchResult],
    scraper_key: str,
    session_id: str,
    emitter: EventEmitter | None = None,
    max_scraped_content: int = 10,
) -> tuple[list[ScrapedSearchResult], list[ImageUrl]]:
    """
    Scrapes the urls
    Args:
        urls: List of urls
        cfg: Config (optional)

    Returns:
        tuple[list[ScrapedContent], ç]: tuple containing scraped content and images

    """
    scraped_data: list[ScrapedSearchResult] = []
    images = []
    try:
        scraper = ScraperRunner(search_results, scraper_key, session_id, max_scraped_content)
        if emitter is not None:
            emitter.forward_events_from(scraper)
        scraped_data = await scraper.run()
        for item in scraped_data:
            if len(item.image_urls) > 0:
                images.extend(item.image_urls)
    except Exception:
        logger.exception(f"{Fore.RED}Error in scrape_urls: {Style.RESET_ALL}")

    return scraped_data, images
