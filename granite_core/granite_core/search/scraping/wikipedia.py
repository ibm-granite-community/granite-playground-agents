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

from urllib.parse import unquote, urlparse

from httpx import AsyncClient, QueryParams

from granite_core.logging import get_logger
from granite_core.search.scraping.base import AsyncScraper
from granite_core.search.scraping.types import ScrapedContent

logger = get_logger(__name__)


class WikipediaScraper(AsyncScraper):
    wikipedia_api_url = "https://en.wikipedia.org/w/api.php"

    async def ascrape(self, link: str, client: AsyncClient) -> ScrapedContent | None:
        path = urlparse(link).path
        title = unquote(path.split("/wiki/")[1])

        params = QueryParams(
            {"action": "query", "format": "json", "titles": title, "prop": "extracts", "explaintext": 1}
        )

        response = await client.get(self.wikipedia_api_url, timeout=10, params=params)

        if response.status_code == 403:
            logger.exception(f"Error 403 when scraping link {link}")
            return None

        data = response.json()
        page = next(iter(data["query"]["pages"].values()))
        return ScrapedContent(url=link, content=page.get("extract", ""), title=page.get("title", ""))
