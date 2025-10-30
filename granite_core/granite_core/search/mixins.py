# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0


from typing import Any

from granite_core.search.scraping.types import ScrapedSearchResult
from granite_core.search.types import SearchResult


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


class ScrapedSearchResultsMixin:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._scraped_search_results: dict[str, ScrapedSearchResult] = {}
        super().__init__(*args, **kwargs)

    def contains_scraped_search_result(self, url: str) -> bool:
        return url in self._scraped_search_results

    def add_scraped_search_results(self, scraped_search_results: list[ScrapedSearchResult]) -> None:
        for s in scraped_search_results:
            self.add_scraped_search_result(s)

    def add_scraped_search_result(self, scraped_search_result: ScrapedSearchResult) -> None:
        if scraped_search_result.url not in self._scraped_search_results:
            self._scraped_search_results[scraped_search_result.url] = scraped_search_result

    @property
    def scraped_search_results(self) -> list[ScrapedSearchResult]:
        return list(self._scraped_search_results.values())
