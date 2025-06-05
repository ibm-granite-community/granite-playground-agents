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
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from transformers import AutoTokenizer

from granite_chat.config import settings
from granite_chat.logger import get_formatted_logger
from granite_chat.search.compressor import CustomVectorstoreCompressor
from granite_chat.search.embeddings import WatsonxEmbeddings
from granite_chat.search.prompts import SearchPrompts
from granite_chat.search.types import SearchResult, SearchResults
from granite_chat.search.vector_store import ConfigurableVectorStoreWrapper

logger = get_formatted_logger(__name__, logging.INFO)


class SearchAgent:
    def __init__(self, chat_model: ChatModel, worker_pool: WorkerPool) -> None:
        self.chat_model = chat_model
        self.worker_pool = worker_pool

        # GPT Researcher config
        self.cfg = Config()

        # Watsonx is handled separately because gpt-researcher does not support it
        # TODO: PR support for watsonx embeddings to gpt-researcher
        if settings.WATSONX_EMBEDDING_MODEL:
            embedding = WatsonxEmbeddings(model_id=settings.WATSONX_EMBEDDING_MODEL)
        else:
            embedding = Memory(
                embedding_provider=self.cfg.embedding_provider, model=self.cfg.embedding_model
            ).get_embeddings()

        vector_store = InMemoryVectorStore(embedding=embedding)

        if settings.EMBEDDING_HF_TOKENIZER:
            tokenizer = AutoTokenizer.from_pretrained(settings.EMBEDDING_HF_TOKENIZER)
            self.vector_store = ConfigurableVectorStoreWrapper(
                vector_store,
                chunk_size=settings.CHUNK_SIZE,
                chunk_overlap=int(settings.CHUNK_OVERLAP),
                tokenizer=tokenizer,
            )
        else:
            # TODO: Config character chunk size
            self.vector_store = ConfigurableVectorStoreWrapper(
                vector_store, chunk_size=settings.CHUNK_SIZE, chunk_overlap=settings.CHUNK_OVERLAP
            )

        self.retriever = get_retrievers({}, self.cfg)[0]

    async def search(self, messages: list[Message]) -> list[Document]:
        # Generate contextualized search queries
        # TODO: parallelize these
        search_queries = await self._generate_search_queries(messages)
        standalone_msg = await self._generate_standalone(messages)

        logger.info(f'Searching with queries => "{search_queries}"')

        # Perform search
        # raw_results = await asyncio.to_thread(lambda: retriever(query=search_queries[0]).search())
        search_results = await self._perform_web_search(search_queries)
        # Scraping
        # search_results = SearchResults(results=[SearchResult(**r) for r in raw_results])
        # search_results = await self.filter_search_results(query, search_results)

        scraped_content: list[dict] = await self._browse_urls([r.href for r in search_results.results])

        # Load scraped context into vector store
        await asyncio.to_thread(lambda: self.vector_store.load(scraped_content))

        # Query vector store
        vectorstore_compressor = CustomVectorstoreCompressor(
            self.vector_store,
            vector_store_filter=None,
            prompt_family=PromptFamily(config=self.cfg),
            **{},  # kwargs
        )

        logger.info(f'Searching for context => "{standalone_msg}"')
        docs: list[Document] = await vectorstore_compressor.async_get_context_docs(query=standalone_msg, max_results=10)

        # Add title from search results to docs
        url_to_result: dict[str, SearchResult] = {result.url: result for result in search_results.results}

        # Add search engine metadata
        for d in docs:
            d.metadata["url"] = url_to_result[d.metadata["source"]].url
            d.metadata["title"] = url_to_result[d.metadata["source"]].title
            d.metadata["snippet"] = url_to_result[d.metadata["source"]].body

        docs = await self._filter_docs(standalone_msg, docs)

        return docs

    async def _browse_urls(self, urls: list[str]) -> list[dict]:
        scraped_content, _ = await scrape_urls(urls, self.cfg, self.worker_pool)
        return scraped_content

    async def _generate_search_queries(self, messages: list[Message]) -> list[str]:
        search_query_prompt = SearchPrompts.generate_search_queries_prompt(messages)
        response = await self.chat_model.create(messages=[UserMessage(content=search_query_prompt)])
        queries = ast.literal_eval(response.get_text_content().strip())
        return queries

    async def _generate_standalone(self, messages: list[Message]) -> str:
        standalone_prompt = SearchPrompts.generate_standalone_query(messages)
        response = await self.chat_model.create(messages=[UserMessage(content=standalone_prompt)])
        return response.get_text_content()

    async def _filter_search_results(self, query: str, search_results: SearchResults) -> SearchResults:
        filtered_results = await asyncio.gather(*(self._filter_search_result(query, r) for r in search_results.results))
        res = [r for r in filtered_results if r is not None]
        return SearchResults(results=res)

    async def _filter_docs(self, query: str, docs: list[Document]) -> list[Document]:
        filtered_docs = await asyncio.gather(*(self._filter_doc(query, d) for d in docs))
        docs = [d for d in filtered_docs if d is not None]
        return docs

    async def _filter_doc(self, query: str, doc: Document) -> Document | None:
        async with self.worker_pool.throttle():
            try:
                filter_context_prompt = SearchPrompts.filter_doc_prompt(query=query, doc=doc)
                response = await self.chat_model.create(messages=[UserMessage(content=filter_context_prompt)])

                if "irrelevant" in response.get_text_content().lower():
                    # logger.info(f"Rejected document {doc.model_dump_json()}")
                    return None

                logger.info(f"Accepted document {doc.model_dump_json()}")
                return doc
            except Exception:
                traceback.print_exc()

        return None

    async def _filter_search_result(self, query: str, search_result: SearchResult) -> SearchResult | None:
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

    async def _perform_web_search(self, queries: list[str], max_results: int = 3) -> SearchResults:
        results = await asyncio.gather(*(self._search_query(q, max_results) for q in queries))
        flat = [item for sublist in results for item in (sublist or [])]

        url_to_result: dict[str, SearchResult] = {result.url: result for result in flat}
        return SearchResults(results=list(url_to_result.values()))

    async def _search_query(self, query: str, max_results: int = 3) -> list[SearchResult] | None:
        async with self.worker_pool.throttle():
            try:
                results = await asyncio.to_thread(lambda: self.retriever(query=query).search(max_results=max_results))
                return [SearchResult(**r) for r in results]
            except Exception:
                traceback.print_exc()
        return None
