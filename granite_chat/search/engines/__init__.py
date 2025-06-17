from granite_chat.search.engines.engine import SearchEngine
from granite_chat.search.engines.google import GoogleSearch


def get_search_engine(provider: str) -> SearchEngine:
    """Return search engine based on provider"""
    if provider == "google":
        return GoogleSearch()
    else:
        raise Exception(f"Unsupported search provider {provider}")
