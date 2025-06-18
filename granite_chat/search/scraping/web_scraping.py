# Portions of this file are derived from the Apache 2.0 licensed project "gpt-researcher"
# Original source: https://github.com/assafelovic/gpt-researcher
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Changes made:

import logging
from typing import Any

from colorama import Fore, Style  # type: ignore

from granite_chat.logger import get_formatted_logger
from granite_chat.search.scraping.extract import ContentExtractor
from granite_chat.workers import WorkerPool

logger = get_formatted_logger(__name__, logging.INFO)


async def scrape_urls(
    urls: list[str], scraper: str, worker_pool: WorkerPool
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Scrapes the urls
    Args:
        urls: List of urls
        cfg: Config (optional)

    Returns:
        tuple[list[dict[str, Any]], list[dict[str, Any]]]: tuple containing scraped content and images

    """
    scraped_data = []
    images = []
    user_agent = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"  # noqa: E501

    try:
        extractor = ContentExtractor(urls, user_agent, scraper, worker_pool=worker_pool)
        scraped_data = await extractor.run()
        for item in scraped_data:
            if "image_urls" in item:
                images.extend(item["image_urls"])
    except Exception as e:
        print(f"{Fore.RED}Error in scrape_urls: {e}{Style.RESET_ALL}")

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
