import asyncio

from beeai_framework.backend import UserMessage
from beeai_framework.backend.chat import ChatModel

from granite_chat import get_logger
from granite_chat.search.prompts import SearchPrompts
from granite_chat.search.types import SearchResult, SearchResultRelevanceSchema
from granite_chat.workers import WorkerPool

logger = get_logger(__name__)


class SearchResultsFilter:
    def __init__(self, chat_model: ChatModel, worker_pool: WorkerPool) -> None:
        self.chat_model = chat_model
        self.worker_pool = worker_pool

    async def filter(self, query: str, results: list[SearchResult]) -> list[SearchResult]:
        filtered_results = await asyncio.gather(*(self._filter_search_result(query, result) for result in results))
        return [r for r in filtered_results if r is not None]

    async def _filter_search_result(self, query: str, result: SearchResult) -> SearchResult | None:
        async with self.worker_pool.throttle():
            prompt = SearchPrompts.filter_search_result_prompt(query=query, search_result=result)
            response = await self.chat_model.create_structure(
                schema=SearchResultRelevanceSchema, messages=[UserMessage(content=prompt)]
            )
            relevance = SearchResultRelevanceSchema(**response.object)
            if relevance.is_relevant:
                return result

            logger.info(f"!! Rejected search result {result.url} {result.title}")
            return None
