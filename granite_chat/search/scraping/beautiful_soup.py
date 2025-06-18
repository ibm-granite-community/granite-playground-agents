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

from granite_chat.search.scraping.scraper import AsyncScraper
from granite_chat.search.scraping.utils import clean_soup, extract_title, get_relevant_images, get_text_from_soup


class BeautifulSoupScraper(AsyncScraper):
    async def ascrape(self, link: str, client: AsyncClient) -> tuple:
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
            response = await client.get(link, timeout=4)
            soup = BeautifulSoup(response.content, "lxml", from_encoding=response.encoding)

            soup = clean_soup(soup)

            content = get_text_from_soup(soup)

            image_urls = get_relevant_images(soup, link)

            # Extract the title using the utility function
            title = extract_title(soup)

            return content, image_urls, title

        except Exception as e:
            print("Error! : " + str(e))
            return "", [], ""
