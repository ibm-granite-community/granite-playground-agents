import asyncio

from beeai_framework.backend import UserMessage

from granite_chat import get_logger
from granite_chat.chat import ChatModelService
from granite_chat.search.prompts import SearchPrompts
from granite_chat.search.types import SearchResult, SearchResultRelevanceSchema

logger = get_logger(__name__)


class SearchResultsFilter:
    def __init__(self, chat_model: ChatModelService) -> None:
        self.chat_model = chat_model

    async def filter(self, query: str, results: list[SearchResult]) -> list[SearchResult]:
        filtered_results = await asyncio.gather(*(self._filter_search_result(query, result) for result in results))
        return [r for r in filtered_results if r is not None]

    async def _filter_search_result(self, query: str, result: SearchResult) -> SearchResult | None:
        prompt = SearchPrompts.filter_search_result_prompt(query=query, search_result=result)
        response = await self.chat_model.create_structure(
            schema=SearchResultRelevanceSchema, messages=[UserMessage(content=prompt)]
        )
        relevance = SearchResultRelevanceSchema(**response.object)
        if relevance.is_relevant:
            return result

        logger.info(f"!! Rejected search result {result.url} {result.title}")
        return None
