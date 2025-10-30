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


from bs4 import BeautifulSoup
from httpx import AsyncClient

from granite_core.logging import get_logger
from granite_core.search.scraping.base import AsyncScraper
from granite_core.search.scraping.types import ScrapedContent
from granite_core.search.scraping.utils import clean_soup, extract_title, get_text_from_soup

logger = get_logger(__name__)


class BeautifulSoupScraper(AsyncScraper):
    async def ascrape(self, link: str, client: AsyncClient) -> ScrapedContent | None:
        """
        This function scrapes content from a webpage by making a GET request, parsing the HTML using
        BeautifulSoup, and extracting script and style elements before returning the cleaned content.

        Returns:
          The `scrape` method is returning the cleaned and extracted content from the webpage specified
        by the `self.link` attribute. The method fetches the webpage content, removes script and style
        tags, extracts the text content, and returns the cleaned content as a string. If any exception
        occurs during the process, an error message is printed and an empty string is returned.
        """
        try:
            response = await client.get(link, timeout=5)

            if response.status_code == 403:
                logger.exception(f"Error 403 when scraping link {link}")
                return None

            soup = BeautifulSoup(response.content, "lxml", from_encoding=response.encoding)

            soup = clean_soup(soup)
            content = get_text_from_soup(soup)
            # image_urls = get_relevant_images(soup, link)

            # Extract the title using the utility function
            title = extract_title(soup)

            return ScrapedContent(content=content, title=title)

        except Exception as e:
            logger.exception(f"Error! : {e!s} scraping link {link}")
            return None
