import asyncio

from beeai_framework.backend import ChatModel, UserMessage

from granite_chat import get_logger_with_prefix
from granite_chat.config import settings
from granite_chat.search.prompts import SearchPrompts
from granite_chat.search.types import SearchResult, SearchResultRelevanceSchema
from granite_chat.work import chat_pool


class SearchResultsFilter:
    def __init__(self, chat_model: ChatModel, session_id: str) -> None:
        self.chat_model = chat_model
        self.logger = get_logger_with_prefix(__name__, tool_name="SearchResultsFilter", session_id=session_id)

    async def filter(self, query: str, results: list[SearchResult]) -> list[SearchResult]:
        filtered_results = await asyncio.gather(*(self._filter_search_result(query, result) for result in results))
        return [r for r in filtered_results if r is not None]

    async def _filter_search_result(self, query: str, result: SearchResult) -> SearchResult | None:
        self.logger.info(f"Validating search result {result.url}")

        prompt = SearchPrompts.filter_search_result_prompt(query=query, search_result=result)

        async with chat_pool.throttle():
            response = await self.chat_model.create_structure(
                schema=SearchResultRelevanceSchema,
                messages=[UserMessage(content=prompt)],
                max_retries=settings.MAX_RETRIES,
            )

        relevance = SearchResultRelevanceSchema(**response.object)

        if relevance.is_relevant:
            return result

        self.logger.info("==================================================")
        self.logger.info(f"Rejected search result: {result.url}")
        self.logger.info(f"Query: {query}")
        self.logger.info(f"Title: {result.title}")
        self.logger.info(f"Body: {result.body}")

        return None
