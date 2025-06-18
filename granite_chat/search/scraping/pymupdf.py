# Portions of this file are derived from the Apache 2.0 licensed project "gpt-researcher"
# Original source: https://github.com/assafelovic/gpt-researcher
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Changes made:

import os
import tempfile
from urllib.parse import urlparse

from httpx import AsyncClient, TimeoutException
from langchain_community.document_loaders import PyMuPDFLoader

from granite_chat.search.scraping.scraper import AsyncScraper


class PyMuPDFScraper(AsyncScraper):
    def is_url(self, link: str) -> bool:
        """
        Check if the provided `link` is a valid URL.

        Returns:
          bool: True if the link is a valid URL, False otherwise.
        """
        try:
            result = urlparse(link)
            return all([result.scheme, result.netloc])  # Check for valid scheme and network location
        except Exception:
            return False

    async def ascrape(self, link: str, client: AsyncClient) -> tuple:
        """
        The `scrape` function uses PyMuPDFLoader to load a document from the provided link (either URL or local file)
        and returns the document as a string.

        Returns:
          str: A string representation of the loaded document.
        """
        try:
            if self.is_url(link):
                async with client, client.stream("GET", link, timeout=5) as response:
                    response.raise_for_status()

                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                    temp_filename = temp_file.name  # Get the temporary file name
                    async for chunk in response.aiter_bytes(chunk_size=8192):
                        temp_file.write(chunk)  # Write the downloaded content to the temporary file

                loader = PyMuPDFLoader(temp_filename)
                doc = loader.load()
                os.remove(temp_filename)
            else:
                loader = PyMuPDFLoader(link)
                doc = loader.load()

            # Extract the content, image (if any), and title from the document.
            image: list[str] = []
            # Retrieve the content of the first page to minimize embedding costs.
            return doc[0].page_content, image, doc[0].metadata["title"]

        except TimeoutException:
            print(f"Download timed out. Please check the link : {link}")
            return "", [], ""
        except Exception as e:
            print(f"Error loading PDF : {link} {e}")
            return "", [], ""
