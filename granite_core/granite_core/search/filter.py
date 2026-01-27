# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0


import asyncio

from beeai_framework.backend import ChatModel, UserMessage

from granite_core.config import settings
from granite_core.logging import get_logger_with_prefix
from granite_core.search.prompts import SearchPrompts
from granite_core.search.types import SearchResult, SearchResultRelevanceSchema
from granite_core.work import chat_pool


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
            response = await self.chat_model.run(
                [UserMessage(content=prompt)],
                response_format=SearchResultRelevanceSchema,
                max_retries=settings.MAX_RETRIES,
            )

        assert isinstance(response.output_structured, SearchResultRelevanceSchema)
        relevance = response.output_structured

        if relevance.is_relevant:
            return result

        self.logger.info("==================================================")
        self.logger.info(f"Rejected search result: {result.url}")
        self.logger.info(f"Query: {query}")
        self.logger.info(f"Title: {result.title}")
        self.logger.info(f"Snippet: {result.snippet}")
        # self.logger.info(f"Rationale: {relevance.rationale}")

        return None
