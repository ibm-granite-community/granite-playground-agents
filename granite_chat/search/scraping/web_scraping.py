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

from granite_chat import get_logger
from granite_chat.emitter import EventEmitter
from granite_chat.search.scraping.extract import ContentExtractor
from granite_chat.search.types import ImageUrl, ScrapedContent, SearchResult

logger = get_logger(__name__)


async def scrape_urls(
    search_results: list[SearchResult], scraper: str, emitter: EventEmitter | None = None, max_scraped_content: int = 10
) -> tuple[list[ScrapedContent], list[ImageUrl]]:
    """
    Scrapes the urls
    Args:
        urls: List of urls
        cfg: Config (optional)

    Returns:
        tuple[list[ScrapedContent], ç]: tuple containing scraped content and images

    """
    scraped_data = []
    images = []
    user_agent = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"  # noqa: E501

    try:
        extractor = ContentExtractor(search_results, user_agent, scraper, max_scraped_content)
        if emitter is not None:
            emitter.forward_events_from(extractor)
        scraped_data = await extractor.run()
        for item in scraped_data:
            if len(item.image_urls) > 0:
                images.extend(item.image_urls)
    except Exception:
        logger.exception(f"{Fore.RED}Error in scrape_urls: {Style.RESET_ALL}")

    return scraped_data, images
