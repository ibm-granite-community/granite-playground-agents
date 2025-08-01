import asyncio
from typing import Any

from acp_sdk import Message as AcpMessage
from acp_sdk import MessagePart
from beeai_framework.backend import ChatModelNewTokenEvent, Message, UserMessage
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore
from pydantic import ValidationError

from granite_chat import get_logger
from granite_chat.chat import ChatModelService
from granite_chat.citations.citations import CitationGeneratorFactory
from granite_chat.config import settings
from granite_chat.emitter import EventEmitter
from granite_chat.events import CitationEvent, GeneratingCitationsEvent, TextEvent, TrajectoryEvent
from granite_chat.research.prompts import ResearchPrompts
from granite_chat.research.types import ResearchPlanSchema, ResearchQuery, ResearchReport
from granite_chat.search.embeddings.embeddings import EmbeddingsFactory
from granite_chat.search.embeddings.tokenizer import EmbeddingsTokenizer
from granite_chat.search.engines.factory import SearchEngineFactory
from granite_chat.search.filter import SearchResultsFilter
from granite_chat.search.mixins import SearchResultsMixin
from granite_chat.search.scraping.web_scraping import scrape_urls
from granite_chat.search.types import SearchResult
from granite_chat.search.vector_store import ConfigurableVectorStoreWrapper


class Researcher(EventEmitter, SearchResultsMixin):
    def __init__(self, chat_model: ChatModelService, messages: list[Message], *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.chat_model = chat_model
        self.messages = messages
        self.logger = get_logger(__name__)

        self.research_topic: str | None = None

        self.research_plan: list[ResearchQuery] = []
        self.interim_reports: list[ResearchReport] = []
        self.final_report_docs: list[Document] = []
        self.final_report: str | None = None

        self.logger.debug("Initializing Researcher")

        embeddings: Embeddings = EmbeddingsFactory.create()
        vector_store = InMemoryVectorStore(embedding=embeddings)
        tokenizer = EmbeddingsTokenizer.get_instance().get_tokenizer()

        if tokenizer:
            self.vector_store = ConfigurableVectorStoreWrapper(
                vector_store=vector_store,
                chunk_size=settings.CHUNK_SIZE - 2,  # minus start/end tokens
                chunk_overlap=int(settings.CHUNK_OVERLAP),
                tokenizer=tokenizer,
            )
        else:
            # Fall back on character chunks
            self.logger.warning("Falling back to vector store without tokenizer")
            self.vector_store = ConfigurableVectorStoreWrapper(
                vector_store=vector_store,
                chunk_size=settings.CHUNK_SIZE,
                chunk_overlap=settings.CHUNK_OVERLAP,
            )

        self.search_results_filter = SearchResultsFilter(chat_model=chat_model)

    async def run(self) -> None:
        """Perform research investigation"""
        self.logger.info("Running Researcher")

        self.research_topic = await self._generate_research_topic()

        await self._emit(TrajectoryEvent(step=f"🚀 Starting research task for '{self.research_topic}'"))
        await self._emit(TrajectoryEvent(step="🤔 Planning research"))

        self.research_plan = await self._generate_research_plan()

        queries = [s.query for s in self.research_plan]
        await self._emit(
            TrajectoryEvent(
                step=f"I will research {queries}",
            )
        )

        self.logger.debug(f"Research plan: {self.research_plan}")

        await self._gather_sources()

        # await asyncio.gather(
        #     *[self._emit(TrajectoryEvent(step=f"🔗 Added source {res.url}")) for res in self.search_results]
        # )

        await self._extract_sources()
        await self._perform_research()

        self.logger.debug(f"reports: {self.interim_reports}")

        self.logger.debug("Starting report writing")
        await self._generate_final_report()

        self.logger.debug("Generating citations")
        await self._generate_citations()

    async def _generate_research_topic(self) -> str:
        """Generate/extract the research topic"""
        return self.messages[-1].text

    async def _gather_sources(self) -> None:
        """Gather all information sources for the research plan"""

        if self.research_plan is None or len(self.research_plan) == 0:
            raise ValueError("No research plan has been set!")

        await asyncio.gather(*(self._gather_sources_for_step(step) for step in self.research_plan))

    async def _gather_sources_for_step(self, query: ResearchQuery) -> None:
        """Gather information for a single research plan step"""

        search_results = await self._search_query(
            query.query, max_results=settings.RESEARCH_MAX_SEARCH_RESULTS_PER_STEP
        )

        search_results = await self.search_results_filter.filter(query.query, search_results)

        for s in search_results:
            self.add_search_result(s)

    async def _extract_sources(self) -> None:
        """Extract all gathered sources"""

        scraped_content, _ = await scrape_urls(search_results=self.search_results, scraper="bs", emitter=self)

        await self._emit(TrajectoryEvent(step="🧑‍💻 Extracting knowledge..."))

        if scraped_content and len(scraped_content) > 0:
            await self.vector_store.load(scraped_content)

    async def _generate_final_report(self) -> None:
        if self.research_topic is None:
            raise ValueError("No research topic set!")

        if self.research_plan is None:
            raise ValueError("No research plan set!")

        if len(self.interim_reports) == 0:
            raise ValueError("No interim reports available!")

        await self._emit(TrajectoryEvent(step="🧠 Generating final report..."))

        prompt = ResearchPrompts.final_report_prompt(topic=self.research_topic, reports=self.interim_reports)

        response: str = ""

        if settings.STREAMING is True:
            # Final report is streamed
            async for data in self.chat_model.create_stream(messages=[UserMessage(content=prompt)]):
                match data:
                    case ChatModelNewTokenEvent():
                        response += data.value.get_text_content()
                        await self._emit(TextEvent(text=data.value.get_text_content()))

        else:
            output = await self.chat_model.create(messages=[UserMessage(content=prompt)])
            response += output.get_text_content()
            await self._emit(TextEvent(text=response))

        self.final_report = response

    async def _generate_research_plan(self) -> list[ResearchQuery]:
        if self.research_topic is None:
            raise ValueError("Research topic has not been set!")

        prompt = ResearchPrompts.research_plan_prompt(
            topic=self.research_topic, max_queries=settings.RESEARCH_PLAN_BREADTH
        )
        response = await self.chat_model.create_structure(
            schema=ResearchPlanSchema, messages=[UserMessage(content=prompt)]
        )

        try:
            plan = ResearchPlanSchema(**response.object)
            return plan.queries
        except ValidationError as e:
            raise ValueError("Failed to generate a valid research plan!") from e

    async def _perform_research(self) -> None:
        if self.research_plan is None or len(self.research_plan) == 0:
            raise ValueError("No research plan has been set!")

        reports = await asyncio.gather(*(self._research_step(step) for step in self.research_plan))
        self.interim_reports.extend(reports)

    async def _research_step(self, query: ResearchQuery) -> ResearchReport:
        await self._emit(TrajectoryEvent(step=f"🧠 Researching '{query.query}'"))

        docs: list[Document] = await self.vector_store.asimilarity_search(
            query=query.query, k=settings.RESEARCH_MAX_DOCS_PER_STEP
        )

        self.final_report_docs += docs

        research_report_prompt = ResearchPrompts.research_report_prompt(query=query, docs=docs)
        response = await self.chat_model.create(messages=[UserMessage(content=research_report_prompt)])
        report = response.get_text_content()
        return ResearchReport(query=query, report=report)

    async def _search_query(self, query: str, max_results: int = 3) -> list[SearchResult]:
        try:
            engine = SearchEngineFactory.create()
            return await engine.search(query=query, max_results=max_results)
        except Exception as e:
            self.logger.exception(repr(e))
        return []

    async def _generate_citations(self) -> None:
        if len(self.final_report_docs) > 0:
            # Compress docs
            docs = self._dedup_documents_by_content(self.final_report_docs)

            input = [AcpMessage(role="user", parts=[MessagePart(name="User", content=self.research_topic)])]

            generator = CitationGeneratorFactory.create()

            await self._emit(GeneratingCitationsEvent())

            async for citation in generator.generate(messages=input, docs=docs, response=self.final_report or ""):
                # yield message_part
                await self._emit(CitationEvent(citation=citation))

    def _dedup_documents_by_content(self, documents: list[Document]) -> list[Document]:
        seen = set()
        deduped = []
        for doc in documents:
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                deduped.append(doc)
        return deduped
