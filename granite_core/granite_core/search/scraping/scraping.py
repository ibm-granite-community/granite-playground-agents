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

from granite_core.config import settings
from granite_core.emitter import EventEmitter
from granite_core.logging import get_logger
from granite_core.search.scraping.runner import ScraperRunner
from granite_core.search.scraping.types import ScrapedContent, ScrapedSearchResult
from granite_core.search.types import SearchResult

logger = get_logger(__name__)


async def scrape_search_results(
    search_results: list[SearchResult],
    scraper_key: str,
    session_id: str = "",
    emitter: EventEmitter | None = None,
    max_scraped_content: int = 10,
) -> list[ScrapedSearchResult]:
    url_map = {s.url: s for s in search_results}

    scraped_contents: list[ScrapedContent] = []

    try:
        scraper = ScraperRunner(list(url_map.keys()), scraper_key, session_id, max_scraped_content)
        if emitter is not None:
            emitter.forward_events_from(scraper)

        scraped_contents = await scraper.run()

    except Exception:
        logger.exception(f"{Fore.RED}Error in scrape_urls: {Style.RESET_ALL}")
    finally:
        await scraper.close()

    return [
        ScrapedSearchResult(
            search_result=url_map[sc.url],
            url=sc.url,
            raw_content=sc.content[: settings.SCRAPER_MAX_CONTENT_LENGTH],  # Trim
            title=sc.title or "",
        )
        for sc in scraped_contents
    ]
