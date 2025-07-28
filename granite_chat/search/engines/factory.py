from granite_chat.config import settings
from granite_chat.search.engines.engine import SearchEngine
from granite_chat.search.engines.google import GoogleSearch
from granite_chat.search.engines.tavily import TavilySearch


class SearchEngineFactory:
    """Factory for ChatModel instances."""

    @staticmethod
    def create() -> SearchEngine:
        provider = settings.RETRIEVER

        if provider == "google":
            return GoogleSearch()
        if provider == "tavily":
            return TavilySearch()
        else:
            raise Exception(f"Unsupported search provider {provider}")
