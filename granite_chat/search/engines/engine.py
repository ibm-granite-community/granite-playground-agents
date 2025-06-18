from abc import ABC, abstractmethod


class SearchEngine(ABC):
    @abstractmethod
    async def search(self, query: str, domains: list[str] | None = None, max_results: int = 7) -> list[dict[str, str]]:
        """Do search"""
        pass
