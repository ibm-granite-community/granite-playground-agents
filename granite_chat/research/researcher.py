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
from granite_chat.research.types import ResearchEvent, ResearchPlanSchema, ResearchReport
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

        self.research_topic: str | None = None
        self.research_plan: list[str] = []
        self.interim_reports: list[ResearchReport] = []

    async def run(self) -> None:
        """Perform research investigation"""
        self.logger.info("Running Researcher")

        self.research_topic = await self._generate_research_topic()
        self.research_plan = await self._generate_research_plan()

        self.logger.debug(f"Research plan: {self.research_plan}")

        await self._gather_sources()
        await self._perform_research()

        self.logger.debug(f"reports: {self.interim_reports}")

        self.logger.debug("Starting report writing")
        await self._generate_final_report()

    async def _generate_research_topic(self) -> str:
        """Generate/extract the research topic"""
        return self.messages[0].text

    async def _gather_sources(self) -> None:
        """Gather all information sources for the research plan"""

        if self.research_plan is None or len(self.research_plan) == 0:
            raise ValueError("No research plan has been set!")

        await asyncio.gather(*(self._gather_sources_for_step(step) for step in self.research_plan))

    async def _gather_sources_for_step(self, plan_step: str) -> None:
        """Gather information for a single research plan step"""
        search_results = await self._search_query(plan_step, max_results=settings.RESEARCH_MAX_SEARCH_RESULTS_PER_STEP)

        if search_results and len(search_results) > 0:
            await self.listener(
                ResearchEvent(
                    event_type="log",
                    data=f"🌐 Found {len(search_results)!s} search results for sub-topic '{plan_step}'",
                )
            )

            scraped_content, _ = await scrape_urls(
                urls=[r.href for r in search_results], scraper="bs", worker_pool=self.worker_pool
            )

            if scraped_content and len(scraped_content) > 0:
                await asyncio.to_thread(lambda: self.vector_store.load(scraped_content))

    async def _generate_final_report(self) -> None:
        if self.research_topic is None:
            raise ValueError("No research topic set!")

        if self.research_plan is None:
            raise ValueError("No research plan set!")

        if len(self.interim_reports) == 0:
            raise ValueError("No interim reports available!")

        await self.listener(ResearchEvent(event_type="log", data="🧠 Generating final report!"))

        prompt = ResearchPrompts.final_report_prompt(
            topic=self.research_topic, plan=self.research_plan, reports=self.interim_reports
        )

        # Final report is streamed
        async for data, event in self.chat_model.create(messages=[UserMessage(content=prompt)], stream=True):
            match (data, event.name):
                case (ChatModelNewTokenEvent(), "new_token"):
                    await self.listener(ResearchEvent(event_type="token", data=data.value.get_text_content()))

    async def _generate_research_plan(self) -> list[str]:
        if self.research_topic is None:
            raise ValueError("Research topic has not been set!")

        await self.listener(ResearchEvent(event_type="log", data="📝 Creating a research plan..."))

        prompt = ResearchPrompts.research_plan_prompt(
            topic=self.research_topic, max_queries=settings.RESEARCH_PLAN_BREADTH
        )
        response = await self.chat_model.create_structure(
            schema=ResearchPlanSchema, messages=[UserMessage(content=prompt)]
        )

        if "plan" in response.object:
            return response.object["plan"]
        else:
            raise ValueError("Failed to generate a valid research plan!")

    async def _perform_research(self) -> None:
        if self.research_plan is None or len(self.research_plan) == 0:
            raise ValueError("No research plan has been set!")

        await asyncio.gather(*(self._research_step(step) for step in self.research_plan))

    async def _research_step(self, step: str) -> None:
        await self.listener(ResearchEvent(event_type="log", data=f"🔍 Researching plan step '{step}'"))

        docs: list[Document] = await self.vector_store.asimilarity_search(
            query=step, k=settings.RESEARCH_MAX_DOCS_PER_STEP
        )

        await self.listener(
            ResearchEvent(event_type="log", data=f"🧠 Generating intermediate report for step '{step}'")
        )

        research_report_prompt = ResearchPrompts.research_report_prompt(topic=step, docs=docs)
        response = await self.chat_model.create(messages=[UserMessage(content=research_report_prompt)])
        report = response.get_text_content()
        self.interim_reports.append(ResearchReport(topic=step, report=report))

    async def _search_query(self, query: str, max_results: int = 3) -> list[SearchResult] | None:
        async with self.worker_pool.throttle():
            try:
                engine = get_search_engine(settings.RETRIEVER)
                results = await engine.search(query=query, max_results=max_results)
                return [SearchResult(**r) for r in results]
            except Exception:
                traceback.print_exc()
        return None
