# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0


import asyncio
from typing import Any

from beeai_framework.backend import ChatModel, Message, UserMessage
from langchain_core.documents import Document

from granite_core.config import settings
from granite_core.logging import get_logger_with_prefix
from granite_core.search.engines.factory import SearchEngineFactory
from granite_core.search.filter import SearchResultsFilter
from granite_core.search.mixins import ScrapedSearchResultsMixin, SearchResultsMixin
from granite_core.search.prompts import SearchPrompts
from granite_core.search.scraping import scrape_search_results
from granite_core.search.types import SearchQueriesSchema, SearchResult, StandaloneQuerySchema
from granite_core.search.vector_store.factory import VectorStoreWrapperFactory
from granite_core.work import chat_pool, task_pool


class SearchTool(SearchResultsMixin, ScrapedSearchResultsMixin):
    def __init__(self, chat_model: ChatModel, session_id: str, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.chat_model = chat_model
        self.vector_store = VectorStoreWrapperFactory.create()

        self.llmaaj_search_filter = SearchResultsFilter(chat_model=self.chat_model, session_id=session_id)

        self.logger = get_logger_with_prefix(__name__, "SearchTool", session_id)
        self.session_id = session_id

    async def search(self, messages: list[Message]) -> list[Document]:
        # Generate contextualized search queries

        search_queries, standalone_msg = await asyncio.gather(
            self._generate_search_queries(messages), self._generate_standalone(messages)
        )

        self.logger.info(f'Searching with queries => "{search_queries}"')

        # Perform search
        await self._perform_web_search(search_queries, max_results=settings.SEARCH_MAX_SEARCH_RESULTS_PER_STEP)
        # Scraping
        await self._browse_urls(self.search_results)

        # Load scraped context into vector store
        await self.vector_store.load(self.scraped_search_results)

        self.logger.info(f'Searching for context => "{standalone_msg}"')

        docs: list[Document] = await self.vector_store.asimilarity_search(
            query=standalone_msg, k=settings.SEARCH_MAX_DOCS_PER_STEP
        )

        return docs

    async def _browse_urls(self, search_results: list[SearchResult]) -> None:
        scraped_results, _ = await scrape_search_results(
            search_results=search_results,
            scraper_key="bs",
            session_id=self.session_id,
            max_scraped_content=settings.SEARCH_MAX_SCRAPED_CONTENT,
        )
        self.add_scraped_search_results(scraped_results)

    async def _generate_search_queries(self, messages: list[Message]) -> list[str]:
        search_query_prompt = SearchPrompts.generate_search_queries_prompt(
            messages, max_queries=settings.SEARCH_MAX_SEARCH_QUERIES_PER_STEP
        )

        async with chat_pool.throttle():
            response = await self.chat_model.create_structure(
                schema=SearchQueriesSchema,
                messages=[UserMessage(content=search_query_prompt)],
                max_retries=settings.MAX_RETRIES,
            )

        result = SearchQueriesSchema(**response.object)
        return result.search_queries[: settings.SEARCH_MAX_SEARCH_QUERIES_PER_STEP]

    async def _generate_standalone(self, messages: list[Message]) -> str:
        standalone_prompt = SearchPrompts.generate_standalone_query(messages)

        async with chat_pool.throttle():
            response = await self.chat_model.create_structure(
                schema=StandaloneQuerySchema,
                messages=[UserMessage(content=standalone_prompt)],
                max_retries=settings.MAX_RETRIES,
            )
        standalone_query = StandaloneQuerySchema(**response.object)
        return standalone_query.query

    async def _perform_web_search(self, queries: list[str], max_results: int = 3) -> None:
        await asyncio.gather(*(self._search_query(q, max_results) for q in queries))

    async def _search_query(self, query: str, max_results: int = 3) -> None:
        try:
            engine = SearchEngineFactory.create()

            # search engines do not throttle internally
            async with task_pool.throttle():
                results = await engine.search(query=query, max_results=max_results)

            # llmaaj filtering
            results = await self.llmaaj_search_filter.filter(query=query, results=results)

            for r in results:
                self.add_search_result(r)
        except Exception as e:
            self.logger.exception(repr(e))
        return None
