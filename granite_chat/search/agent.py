import asyncio
import traceback

from beeai_framework.backend import Message, UserMessage
from beeai_framework.backend.chat import ChatModel
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore

from granite_chat import get_logger
from granite_chat.config import settings
from granite_chat.search.embeddings import get_embeddings
from granite_chat.search.embeddings.tokenizer import EmbeddingsTokenizer
from granite_chat.search.engines import get_search_engine
from granite_chat.search.prompts import SearchPrompts
from granite_chat.search.scraping.web_scraping import scrape_urls
from granite_chat.search.types import ScrapedContent, SearchQueriesSchema, SearchResult
from granite_chat.search.vector_store import ConfigurableVectorStoreWrapper
from granite_chat.workers import WorkerPool

logger = get_logger(__name__)


class SearchAgent:
    def __init__(self, chat_model: ChatModel, worker_pool: WorkerPool) -> None:
        self.chat_model = chat_model
        self.worker_pool = worker_pool

        embeddings: Embeddings = get_embeddings(
            provider=settings.EMBEDDINGS_PROVIDER, model_name=settings.EMBEDDINGS_MODEL
        )

        vector_store = InMemoryVectorStore(embedding=embeddings)

        if settings.EMBEDDINGS_HF_TOKENIZER and (tokenizer := EmbeddingsTokenizer.get_instance().get_tokenizer()):
            self.vector_store = ConfigurableVectorStoreWrapper(
                vector_store,
                chunk_size=settings.CHUNK_SIZE - 2,  # minus start/end tokens
                chunk_overlap=int(settings.CHUNK_OVERLAP),
                tokenizer=tokenizer,
            )
        else:
            # Fall back on character chunks
            self.vector_store = ConfigurableVectorStoreWrapper(
                vector_store, chunk_size=settings.CHUNK_SIZE, chunk_overlap=settings.CHUNK_OVERLAP
            )

        self.search_engine = get_search_engine(settings.RETRIEVER)

    async def search(self, messages: list[Message]) -> list[Document]:
        # Generate contextualized search queries
        # TODO: parallelize these
        search_queries = await self._generate_search_queries(messages)
        standalone_msg = await self._generate_standalone(messages)

        logger.info(f'Searching with queries => "{search_queries}"')

        # Perform search
        search_results = await self._perform_web_search(search_queries)
        # Scraping
        scraped_content = await self._browse_urls(search_results)

        # Load scraped context into vector store
        await asyncio.to_thread(lambda: self.vector_store.load(scraped_content))

        logger.info(f'Searching for context => "{standalone_msg}"')

        docs: list[Document] = await self.vector_store.asimilarity_search(query=standalone_msg, k=10)

        return docs

    async def _browse_urls(self, search_results: list[SearchResult]) -> list[ScrapedContent]:
        scraped_content, _ = await scrape_urls(
            search_results=search_results, scraper="bs", worker_pool=self.worker_pool
        )
        return scraped_content

    async def _generate_search_queries(self, messages: list[Message]) -> list[str]:
        search_query_prompt = SearchPrompts.generate_search_queries_prompt(messages)
        response = await self.chat_model.create_structure(
            schema=SearchQueriesSchema, messages=[UserMessage(content=search_query_prompt)]
        )
        if "search_queries" in response.object:
            return response.object["search_queries"]
        else:
            raise ValueError("Failed to generate valid search queries!")

    async def _generate_standalone(self, messages: list[Message]) -> str:
        standalone_prompt = SearchPrompts.generate_standalone_query(messages)
        response = await self.chat_model.create(messages=[UserMessage(content=standalone_prompt)])
        return response.get_text_content()

    async def _perform_web_search(self, queries: list[str], max_results: int = 3) -> list[SearchResult]:
        results = await asyncio.gather(*(self._search_query(q, max_results) for q in queries))
        flat = [item for sublist in results if sublist is not None for item in sublist]
        return flat

    async def _search_query(self, query: str, max_results: int = 3) -> list[SearchResult] | None:
        async with self.worker_pool.throttle():
            try:
                engine = get_search_engine(settings.RETRIEVER)
                results = await engine.search(query=query, max_results=max_results)
                return [SearchResult(**r) for r in results]
            except Exception:
                traceback.print_exc()
        return None
