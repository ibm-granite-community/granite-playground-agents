from typing import Any

from granite_chat.search.types import SearchResult


class SearchResultsMixin:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._search_results: dict[str, SearchResult] = {}
        super().__init__(*args, **kwargs)

    def add_search_result(self, search_result: SearchResult) -> None:
        if search_result.url not in self.search_results:
            self._search_results[search_result.url] = search_result

    @property
    def search_results(self) -> list[SearchResult]:
        return list(self._search_results.values())
