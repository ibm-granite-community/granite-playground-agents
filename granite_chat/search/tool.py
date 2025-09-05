import asyncio
from typing import Any

from beeai_framework.backend import ChatModel, Message, UserMessage
from langchain_core.documents import Document

from granite_chat import get_logger
from granite_chat.config import settings
from granite_chat.search.engines.factory import SearchEngineFactory
from granite_chat.search.filter import SearchResultsFilter
from granite_chat.search.mixins import SearchResultsMixin
from granite_chat.search.prompts import SearchPrompts
from granite_chat.search.scraping.web_scraping import scrape_urls
from granite_chat.search.types import ScrapedContent, SearchQueriesSchema, SearchResult, StandaloneQuerySchema
from granite_chat.search.vector_store.factory import VectorStoreWrapperFactory
from granite_chat.work import chat_pool, task_pool

logger = get_logger(__name__)


class SearchTool(SearchResultsMixin):
    def __init__(self, chat_model: ChatModel, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.chat_model = chat_model
        self.vector_store = VectorStoreWrapperFactory.create()
        self.search_results_filter = SearchResultsFilter(chat_model=self.chat_model)

    async def search(self, messages: list[Message]) -> list[Document]:
        # Generate contextualized search queries

        search_queries, standalone_msg = await asyncio.gather(
            self._generate_search_queries(messages), self._generate_standalone(messages)
        )

        logger.info(f'Searching with queries => "{search_queries}"')

        # Perform search
        await self._perform_web_search(search_queries, max_results=settings.SEARCH_MAX_SEARCH_RESULTS_PER_STEP)
        # Scraping
        scraped_content = await self._browse_urls(self.search_results)

        # Load scraped context into vector store
        await self.vector_store.load(scraped_content)

        logger.info(f'Searching for context => "{standalone_msg}"')

        docs: list[Document] = await self.vector_store.asimilarity_search(
            query=standalone_msg, k=settings.SEARCH_MAX_DOCS_PER_STEP
        )

        return docs

    async def _browse_urls(self, search_results: list[SearchResult]) -> list[ScrapedContent]:
        scraped_content, _ = await scrape_urls(search_results=search_results, scraper="bs")
        return scraped_content

    async def _generate_search_queries(self, messages: list[Message]) -> list[str]:
        search_query_prompt = SearchPrompts.generate_search_queries_prompt(messages)

        async with chat_pool.throttle():
            response = await self.chat_model.create_structure(
                schema=SearchQueriesSchema, messages=[UserMessage(content=search_query_prompt)]
            )
        if "search_queries" in response.object:
            return response.object["search_queries"]
        else:
            raise ValueError("Failed to generate valid search queries!")

    async def _generate_standalone(self, messages: list[Message]) -> str:
        standalone_prompt = SearchPrompts.generate_standalone_query(messages)

        async with chat_pool.throttle():
            response = await self.chat_model.create_structure(
                schema=StandaloneQuerySchema, messages=[UserMessage(content=standalone_prompt)]
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

            results = await self.search_results_filter.filter(query=query, results=results)
            for r in results:
                self.add_search_result(r)
        except Exception as e:
            logger.exception(repr(e))
        return None
