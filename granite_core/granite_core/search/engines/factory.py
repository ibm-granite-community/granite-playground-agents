from granite_core.config import settings
from granite_core.search.engines.engine import SearchEngine
from granite_core.search.engines.google import GoogleSearch
from granite_core.search.engines.tavily import TavilySearch


class SearchEngineFactory:
    """Factory for SearchEngine instances."""

    @staticmethod
    def create() -> SearchEngine:
        provider = settings.RETRIEVER

        if provider == "google":
            return GoogleSearch()
        if provider == "tavily":
            return TavilySearch()
        else:
            raise Exception(f"Unsupported search provider {provider}")
