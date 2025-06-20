from granite_chat.search.engines.engine import SearchEngine
from granite_chat.search.engines.google import GoogleSearch
from granite_chat.search.engines.tavily import TavilySearch


def get_search_engine(provider: str) -> SearchEngine:
    """Return search engine based on provider"""
    if provider == "google":
        return GoogleSearch()
    if provider == "tavily":
        return TavilySearch()
    else:
        raise Exception(f"Unsupported search provider {provider}")
