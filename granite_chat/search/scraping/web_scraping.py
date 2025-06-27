# Portions of this file are derived from the Apache 2.0 licensed project "gpt-researcher"
# Original source: https://github.com/assafelovic/gpt-researcher
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Changes made:

from typing import Any

from colorama import Fore, Style  # type: ignore

from granite_chat import get_logger
from granite_chat.search.scraping.extract import ContentExtractor
from granite_chat.search.types import ImageUrl, ScrapedContent, SearchResult
from granite_chat.workers import WorkerPool

logger = get_logger(__name__)


async def scrape_urls(
    search_results: list[SearchResult], scraper: str, worker_pool: WorkerPool
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
        unique_results = list({r.url: r for r in search_results}.values())
        extractor = ContentExtractor(unique_results, user_agent, scraper, worker_pool=worker_pool)
        scraped_data = await extractor.run()
        for item in scraped_data:
            if len(item.image_urls) > 0:
                images.extend(item.image_urls)
    except Exception:
        logger.exception(f"{Fore.RED}Error in scrape_urls: {Style.RESET_ALL}")

    return scraped_data, images


async def filter_urls(urls: list[str], excluded_domains: list[str]) -> list[str]:
    """
    Filter URLs based on configuration settings.

    Args:
        urls (list[str]): List of URLs to filter.
        config (Config): Configuration object.

    Returns:
        list[str]: Filtered list of URLs.
    """
    filtered_urls = []
    for url in urls:
        # Add your filtering logic here
        # For example, you might want to exclude certain domains or URL patterns
        if not any(excluded in url for excluded in excluded_domains):
            filtered_urls.append(url)
    return filtered_urls


async def extract_main_content(html_content: str) -> str:
    """
    Extract the main content from HTML.

    Args:
        html_content (str): Raw HTML content.

    Returns:
        str: Extracted main content.
    """
    # Implement content extraction logic here
    # This could involve using libraries like BeautifulSoup or custom parsing logic
    # For now, we'll just return the raw HTML as a placeholder
    return html_content


async def process_scraped_data(scraped_data: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Process the scraped data to extract and clean the main content.

    Args:
        scraped_data (list[dict[str, Any]]): List of dictionaries containing scraped data.
        config (Config): Configuration object.

    Returns:
        list[dict[str, Any]]: Processed scraped data.
    """
    processed_data = []
    for item in scraped_data:
        if item["status"] == "success":
            main_content = await extract_main_content(item["content"])
            processed_data.append({"url": item["url"], "content": main_content, "status": "success"})
        else:
            processed_data.append(item)
    return processed_data
