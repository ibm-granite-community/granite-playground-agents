# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0


from abc import ABC, abstractmethod

from granite_core.search.types import SearchResult


class SearchEngine(ABC):
    @abstractmethod
    async def search(self, query: str, domains: list[str] | None = None, max_results: int = 7) -> list[SearchResult]:
        """Do search"""
        pass
