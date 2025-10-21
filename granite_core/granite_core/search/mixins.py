# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0


from typing import Any

from granite_core.search.types import ScrapedContent, SearchResult


class SearchResultsMixin:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._search_results: dict[str, SearchResult] = {}
        super().__init__(*args, **kwargs)

    def contains_search_result(self, url: str) -> bool:
        return url in self._search_results

    def add_search_results(self, search_results: list[SearchResult]) -> None:
        for s in search_results:
            self.add_search_result(s)

    def add_search_result(self, search_result: SearchResult) -> None:
        if search_result.url not in self.search_results:
            self._search_results[search_result.url] = search_result

    @property
    def search_results(self) -> list[SearchResult]:
        return list(self._search_results.values())


class ScrapedContentMixin:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._scraped_content: dict[str, ScrapedContent] = {}
        super().__init__(*args, **kwargs)

    def contains_scraped_content(self, url: str) -> bool:
        return url in self._scraped_content

    def add_scraped_contents(self, scraped_contents: list[ScrapedContent]) -> None:
        for s in scraped_contents:
            self.add_scraped_content(s)

    def add_scraped_content(self, scraped_content: ScrapedContent) -> None:
        if scraped_content.url not in self._scraped_content:
            self._scraped_content[scraped_content.url] = scraped_content

    @property
    def scraped_contents(self) -> list[ScrapedContent]:
        return list(self._scraped_content.values())
