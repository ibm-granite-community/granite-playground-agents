from ddgs import DDGS

from granite_core.config import settings
from granite_core.logging import get_logger
from granite_core.search.engines.engine import SearchEngine
from granite_core.search.types import SearchResult

logger = get_logger(__name__)


class DuckDuckGoSearch(SearchEngine):
    """
    DuckDuckGo Search engine
    """

    async def search(self, query: str, domains: list[str] | None = None, max_results: int = 7) -> list[SearchResult]:
        results = DDGS(
            proxy=settings.DDG_SEARCH_PROXY,
            verify=settings.DDG_SEARCH_VERIFY,
        ).text(
            query,
            max_results=max_results,
            safesearch="on" if settings.SAFE_SEARCH else "moderate",
        )

        search_results = []

        for result in results:
            if "youtube.com" in result["href"]:
                continue
            try:
                search_result = SearchResult(
                    title=result.get("title", ""),
                    href=result.get("href", ""),
                    body=result.get("body", ""),
                )
            except Exception:
                continue
            search_results.append(search_result)

        return search_results[:max_results]
