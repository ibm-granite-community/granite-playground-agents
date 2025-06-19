import ast
import asyncio
import traceback
from collections.abc import Awaitable, Callable

from beeai_framework.backend import ChatModelNewTokenEvent, Message, UserMessage
from beeai_framework.backend.chat import ChatModel
from beeai_framework.logger import Logger
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore
from transformers import AutoTokenizer

from granite_chat.config import settings
from granite_chat.research.prompts import ResearchPrompts
from granite_chat.research.types import ResearchEvent, ResearchReport
from granite_chat.search.embeddings import get_embeddings
from granite_chat.search.engines import get_search_engine
from granite_chat.search.scraping.web_scraping import scrape_urls
from granite_chat.search.types import SearchResult
from granite_chat.search.vector_store import ConfigurableVectorStoreWrapper
from granite_chat.workers import WorkerPool


class Researcher:
    def __init__(
        self,
        chat_model: ChatModel,
        messages: list[Message],
        worker_pool: WorkerPool,
        listener: Callable[[ResearchEvent], Awaitable[None]],
    ) -> None:
        self.chat_model = chat_model
        self.messages = messages
        self.worker_pool = worker_pool
        self.listener = listener
        self.logger = Logger("Researcher", level="DEBUG")

        self.logger.debug("Initializing Researcher")

        embeddings: Embeddings = get_embeddings(
            provider=settings.EMBEDDINGS_PROVIDER, model_name=settings.EMBEDDINGS_MODEL
        )

        vector_store = InMemoryVectorStore(embedding=embeddings)

        if settings.EMBEDDINGS_HF_TOKENIZER:
            tokenizer = AutoTokenizer.from_pretrained(settings.EMBEDDINGS_HF_TOKENIZER)
            self.vector_store = ConfigurableVectorStoreWrapper(
                vector_store,
                chunk_size=settings.CHUNK_SIZE - 2,  # minus start/end tokens
                chunk_overlap=int(settings.CHUNK_OVERLAP),
                tokenizer=tokenizer,
            )
        else:
            # Fall back on character chunks
            self.logger.warning("Falling back to vector store without tokenizer")
            self.vector_store = ConfigurableVectorStoreWrapper(
                vector_store, chunk_size=settings.CHUNK_SIZE, chunk_overlap=settings.CHUNK_OVERLAP
            )

        self.search_engine = get_search_engine(settings.RETRIEVER)

    async def run(self) -> None:
        """Perform research investigation"""
        self.logger.info("Running Researcher")

        topic = self.messages[0].text

        sub_queries = await self._generate_research_plan(topic=topic)
        self.logger.debug(f"sub_queries: {sub_queries}")

        reports = await self._perform_research(sub_queries)
        self.logger.debug(f"reports: {reports}")

        self.logger.debug("Starting report writing")
        await self._generate_final_report(topic=topic, reports=reports)

    async def _generate_final_report(self, topic: str, reports: list[ResearchReport]) -> None:
        prompt = ResearchPrompts.final_report_prompt(topic=topic, reports=reports)

        async for data, event in self.chat_model.create(messages=[UserMessage(content=prompt)], stream=True):
            match (data, event.name):
                case (ChatModelNewTokenEvent(), "new_token"):
                    await self.listener(ResearchEvent(event_type="token", data=data.value.get_text_content()))

    async def _generate_research_plan(self, topic: str) -> list[str]:
        prompt = ResearchPrompts.research_plan_prompt(topic=topic)
        response = await self.chat_model.create(messages=[UserMessage(content=prompt)])
        queries = ast.literal_eval(response.get_text_content().strip())
        return queries

    async def _perform_research(self, queries: list[str]) -> list[ResearchReport]:
        results = await asyncio.gather(*(self._research_topic(q) for q in queries))
        filtered = [x for x in results if x is not None]
        return filtered

    async def _research_topic(self, query: str) -> ResearchReport | None:
        search_results = await self._search_query(query)

        if search_results:
            scraped_content, _ = await scrape_urls(
                urls=[r.href for r in search_results], scraper="bs", worker_pool=self.worker_pool
            )

            # TODO: Should store be shared like this?
            await asyncio.to_thread(lambda: self.vector_store.load(scraped_content))

            docs: list[Document] = await self.vector_store.asimilarity_search(query=query, k=10)

            # url_to_result: dict[str, SearchResult] = {result.url: result for result in search_results}

            # try:
            #     for d in docs:
            #         if d.metadata["source"] in url_to_result:
            #             d.metadata["url"] = url_to_result[d.metadata["source"]].url
            #             d.metadata["title"] = url_to_result[d.metadata["source"]].title
            #             d.metadata["snippet"] = url_to_result[d.metadata["source"]].body
            # except KeyError as e:
            #     pass

            research_report_prompt = ResearchPrompts.research_report_prompt(topic=query, docs=docs)
            response = await self.chat_model.create(messages=[UserMessage(content=research_report_prompt)])
            report = response.get_text_content()
            return ResearchReport(topic=query, report=report)

        return None

    async def _search_query(self, query: str, max_results: int = 3) -> list[SearchResult] | None:
        async with self.worker_pool.throttle():
            try:
                engine = get_search_engine(settings.RETRIEVER)
                results = await engine.search(query=query, max_results=max_results)
                return [SearchResult(**r) for r in results]
            except Exception:
                traceback.print_exc()
        return None
