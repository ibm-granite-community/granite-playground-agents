import ast
import asyncio

from beeai_framework.backend import Message, UserMessage
from beeai_framework.backend.chat import ChatModel
from gpt_researcher.actions.retriever import get_retrievers  # type: ignore
from gpt_researcher.actions.web_scraping import scrape_urls  # type: ignore
from gpt_researcher.config.config import Config  # type: ignore
from gpt_researcher.memory.embeddings import Memory  # type: ignore
from gpt_researcher.prompts import PromptFamily  # type: ignore
from gpt_researcher.utils.logger import get_formatted_logger  # type: ignore
from gpt_researcher.utils.workers import WorkerPool  # type: ignore
from gpt_researcher.vector_store import VectorStoreWrapper  # type: ignore
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from pydantic import BaseModel

from granite_chat.search.compressor import CustomVectorstoreCompressor
from granite_chat.search.prompts import SearchPrompts

logger = get_formatted_logger()


class SearchResult(BaseModel):
    title: str
    href: str
    body: str


class SearchResults(BaseModel):
    results: list[SearchResult] = []


class SearchAgent:

    def __init__(self, chat_model: ChatModel) -> None:
        self.chat_model = chat_model
        self.cfg = Config()

        # TODO
        self.cfg.retrievers = ["google"]

        memory = Memory(embedding_provider=self.cfg.embedding_provider, model=self.cfg.embedding_model)

        # TODO: Watsonx embeddings
        vector_store = InMemoryVectorStore(embedding=memory.get_embeddings())

        self.vector_store = VectorStoreWrapper(vector_store)
        self.vector_store_filter = None
        self.prompt_family = PromptFamily(config=self.cfg)

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
        scraped_content: list[dict] = await self.browse_urls([r.href for r in search_results.results])

        # Load scraped context into vector store
        await asyncio.to_thread(lambda: self.vector_store.load(scraped_content))

        # Query vector store
        vectorstore_compressor = CustomVectorstoreCompressor(
            self.vector_store,
            self.vector_store_filter,
            prompt_family=self.prompt_family,
            **{},  # kwargs
        )

        docs: list[Document] = await vectorstore_compressor.async_get_context_docs(query=query, max_results=10)
        return docs

    async def browse_urls(self, urls: list[str]) -> list[dict]:
        self.worker_pool = WorkerPool(self.cfg.max_scraper_workers)
        scraped_content, _ = await scrape_urls(urls, self.cfg, self.worker_pool)
        return scraped_content

    async def generate_search_queries(self, messages: list[Message]) -> list[str]:

        search_query_prompt = SearchPrompts.generate_search_queries_prompt(messages)
        response = await self.chat_model.create(messages=[UserMessage(content=search_query_prompt)])
        queries = ast.literal_eval(response.get_text_content().strip())
        return queries
