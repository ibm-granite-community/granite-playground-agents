import ast
import asyncio
import logging
import traceback

from beeai_framework.backend import Message, UserMessage
from beeai_framework.backend.chat import ChatModel
from gpt_researcher.actions.retriever import get_retrievers  # type: ignore
from gpt_researcher.actions.web_scraping import scrape_urls  # type: ignore
from gpt_researcher.config.config import Config  # type: ignore
from gpt_researcher.memory.embeddings import Memory  # type: ignore
from gpt_researcher.prompts import PromptFamily  # type: ignore
from gpt_researcher.utils.workers import WorkerPool  # type: ignore
from gpt_researcher.vector_store import VectorStoreWrapper  # type: ignore
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore

from granite_chat.logger import get_formatted_logger
from granite_chat.search.compressor import CustomVectorstoreCompressor
from granite_chat.search.prompts import SearchPrompts
from granite_chat.search.types import SearchResult, SearchResults

logger = get_formatted_logger(__name__, logging.INFO)


class SearchAgent:
    def __init__(self, chat_model: ChatModel, worker_pool: WorkerPool) -> None:
        self.chat_model = chat_model
        self.worker_pool = worker_pool

        self.cfg = Config()

        memory = Memory(embedding_provider=self.cfg.embedding_provider, model=self.cfg.embedding_model)

        # TODO: Watsonx embeddings
        vector_store = InMemoryVectorStore(embedding=memory.get_embeddings())
        self.vector_store = VectorStoreWrapper(vector_store)

    async def search(self, messages: list[Message]) -> list[Document]:
        # Generate contextualized search queries
        search_queries = await self.generate_search_queries(messages)
        retriever = get_retrievers({}, self.cfg)[0]

        # TODO: Run simultaneous query variants
        query = search_queries[0]

        logger.info(f'Searching with query => "{query}"')

        # Perform search
        raw_results = await asyncio.to_thread(lambda: retriever(query=search_queries[0]).search())

        # Scraping
        search_results = SearchResults(results=[SearchResult(**r) for r in raw_results])
        search_results = await self.filter_search_results(query, search_results)

        scraped_content: list[dict] = await self.browse_urls([r.href for r in search_results.results])

        # Load scraped context into vector store
        await asyncio.to_thread(lambda: self.vector_store.load(scraped_content))

        # Query vector store
        vectorstore_compressor = CustomVectorstoreCompressor(
            self.vector_store,
            vector_store_filter=None,
            prompt_family=PromptFamily(config=self.cfg),
            **{},  # kwargs
        )

        docs: list[Document] = await vectorstore_compressor.async_get_context_docs(query=query, max_results=10)

        # Add title from search results to docs
        url_to_result: dict[str, SearchResult] = {result.url: result for result in search_results.results}

        # Add search engine metadata
        for d in docs:
            d.metadata["url"] = url_to_result[d.metadata["source"]].url
            d.metadata["title"] = url_to_result[d.metadata["source"]].title
            d.metadata["snippet"] = url_to_result[d.metadata["source"]].body

        return docs

    async def browse_urls(self, urls: list[str]) -> list[dict]:
        scraped_content, _ = await scrape_urls(urls, self.cfg, self.worker_pool)
        return scraped_content

    async def generate_search_queries(self, messages: list[Message]) -> list[str]:
        search_query_prompt = SearchPrompts.generate_search_queries_prompt(messages)
        response = await self.chat_model.create(messages=[UserMessage(content=search_query_prompt)])
        queries = ast.literal_eval(response.get_text_content().strip())
        return queries

    async def filter_search_results(self, query: str, search_results: SearchResults) -> SearchResults:
        filtered_results = await asyncio.gather(*(self.filter_search_result(query, r) for r in search_results.results))
        res = [r for r in filtered_results if r is not None]
        return SearchResults(results=res)

    async def filter_search_result(self, query: str, search_result: SearchResult) -> SearchResult | None:
        async with self.worker_pool.throttle():
            try:
                search_filter_prompt = SearchPrompts.filter_search_result_prompt(
                    query=query, search_result=search_result
                )
                response = await self.chat_model.create(messages=[UserMessage(content=search_filter_prompt)])

                if "irrelevant" in response.get_text_content().lower():
                    logger.info(f"Rejected search result {search_result.url}")
                    return None

                logger.info(f"Accepted search result {search_result.url}")
                return search_result
            except Exception:
                traceback.print_exc()

        return None
