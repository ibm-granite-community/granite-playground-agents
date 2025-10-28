# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0


import asyncio
from typing import Any

from beeai_framework.backend import (
    ChatModel,
    ChatModelNewTokenEvent,
    ChatModelSuccessEvent,
    Message,
    Role,
    SystemMessage,
    UserMessage,
)
from langchain_core.documents import Document
from pydantic import ValidationError

from granite_core.citations.citations import CitationGeneratorFactory
from granite_core.config import settings
from granite_core.emitter import EventEmitter
from granite_core.events import (
    GeneratingCitationsCompleteEvent,
    GeneratingCitationsEvent,
    PassThroughEvent,
    TextEvent,
    TrajectoryEvent,
)
from granite_core.logging import get_logger_with_prefix
from granite_core.research.prompts import ResearchPrompts
from granite_core.research.types import (
    LanguageIdentificationSchema,
    ResearchPlanSchema,
    ResearchQuery,
    ResearchReport,
    ResearchTopicSchema,
)
from granite_core.search.engines.factory import SearchEngineFactory
from granite_core.search.filter import SearchResultsFilter
from granite_core.search.mixins import ScrapedContentMixin, SearchResultsMixin
from granite_core.search.prompts import SearchPrompts
from granite_core.search.scraping.web_scraping import scrape_urls
from granite_core.search.tool import SearchTool
from granite_core.search.types import SearchResult
from granite_core.search.vector_store.factory import VectorStoreWrapperFactory
from granite_core.work import chat_pool, task_pool


class Researcher(
    EventEmitter,
    SearchResultsMixin,
    ScrapedContentMixin,
):
    def __init__(
        self,
        chat_model: ChatModel,
        structured_chat_model: ChatModel,
        messages: list[Message],
        session_id: str,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.chat_model = chat_model
        self.structured_chat_model = structured_chat_model or chat_model
        self.messages = messages
        self.session_id = session_id
        self.logger = get_logger_with_prefix(__name__, tool_name="Researcher", session_id=session_id)

        self.research_topic: str | None = None
        self.pre_research: str | None = None

        self.research_plan: list[ResearchQuery] = []
        self.interim_reports: list[ResearchReport] = []
        self.final_report_docs: list[Document] = []
        self.final_report: str | None = None

        self.logger.debug("Initializing Researcher")
        self.vector_store = VectorStoreWrapperFactory.create()
        self.search_results_filter = SearchResultsFilter(chat_model=self.structured_chat_model, session_id=session_id)

    async def run(self) -> None:
        """Perform research investigation"""
        self.logger.info("Running Researcher")

        self.research_topic = await self._generate_research_topic()

        await self._emit(TrajectoryEvent(title="Research topic", content=self.research_topic.strip()))
        # await self._emit(TrajectoryEvent(title="Developing a research plan"))

        # Do some pre research
        await self._emit(TrajectoryEvent(title="Conducting preliminary research"))
        self._context = await self._generate_research_context()

        # Generate the research plan
        self.research_plan = await self._generate_research_plan()

        await self._emit(TrajectoryEvent(title="Research plan", content=[s.question for s in self.research_plan]))

        self.logger.debug(f"Research plan: {self.research_plan}")

        # await self._emit(TrajectoryEvent(title="Gathering information"))

        await self._emit(TrajectoryEvent(title="Searching for information"))
        await self._gather_sources()
        await self._extract_sources()

        # await self._emit(TrajectoryEvent(title="Performing research"))

        if self.scraped_contents:
            await self.vector_store.load(self.scraped_contents)

        await self._perform_research()

        self.logger.debug(f"reports: {self.interim_reports}")

        self.logger.info("Starting report writing")
        await self._generate_final_report()

        self.logger.info("Generating citations")
        await self._generate_citations()
        self.logger.info("Research run complete.")

    async def _generate_research_topic(self) -> str:
        """Generate/extract the research topic"""
        standalone_prompt = ResearchPrompts.interpret_research_topic(self.messages)

        async with chat_pool.throttle():
            response = await self.chat_model.create_structure(
                schema=ResearchTopicSchema,
                messages=[UserMessage(content=standalone_prompt)],
                max_retries=settings.MAX_RETRIES,
            )
        topic = ResearchTopicSchema(**response.object)
        return topic.research_topic

    async def _generate_research_context(self) -> str:
        """
        Conduct Preliminary research
        """
        search_tool = SearchTool(chat_model=self.structured_chat_model, session_id=self.session_id)
        docs: list[Document] = await search_tool.search(self.messages)

        # Merge existing search results and scraped content to avoid duplication
        self.add_search_results(search_tool.search_results)
        self.add_scraped_contents(search_tool.scraped_contents)

        search_messages: list[Message] = [
            SystemMessage(content=SearchPrompts.search_system_prompt(docs, include_core_chat=False))
        ]

        async with chat_pool.throttle():
            output = await self.chat_model.create(
                messages=search_messages,
                max_retries=settings.MAX_RETRIES,
                max_tokens=settings.RESEARCH_PRELIM_MAX_TOKENS,
            )

        return output.get_text_content()

    async def _gather_sources(self) -> None:
        """Gather all information sources for the research plan"""
        if self.research_plan is None or len(self.research_plan) == 0:
            raise ValueError("No research plan has been set!")

        await asyncio.gather(*(self._gather_sources_for_step(step) for step in self.research_plan))

    async def _gather_sources_for_step(self, query: ResearchQuery) -> None:
        """Gather information for a single research plan step"""
        await self._web_search(" ".join([query.question, query.search_query]))

    async def _web_search(self, query: str) -> None:
        search_results = await self._search_query(query, max_results=settings.RESEARCH_MAX_SEARCH_RESULTS_PER_STEP)
        filtered = [s for s in search_results if not self.contains_search_result(s.url)]
        search_results = await self.search_results_filter.filter(query, filtered)
        for s in search_results:
            self.add_search_result(s)

    async def _extract_sources(self) -> None:
        """Extract all gathered sources"""

        filtered = [s for s in self.search_results if self.contains_scraped_content(s.url)]
        scraped_contents, _ = await scrape_urls(
            search_results=filtered,
            scraper="bs",
            session_id=self.session_id,
            emitter=self,
            max_scraped_content=settings.RESEARCH_MAX_SCRAPED_CONTENT,
        )
        self.add_scraped_contents(scraped_contents)
        # await self._emit(TrajectoryEvent(title="Extracting knowledge"))

    def _get_most_recent_user_message(self) -> Message:
        return next((message for message in reversed(self.messages) if message.role == Role.USER), self.messages[0])

    async def _get_language(self) -> str:
        recent_user_message = self._get_most_recent_user_message()
        async with chat_pool.throttle():
            response = await self.structured_chat_model.create_structure(
                schema=LanguageIdentificationSchema,
                messages=[UserMessage(content=ResearchPrompts.language_identification(recent_user_message.text))],
                max_retries=settings.MAX_RETRIES,
            )
            try:
                identified_language = LanguageIdentificationSchema(**response.object).language.title()
                self.logger.info(
                    f"Writing report in {identified_language} based on the most recent user message: {recent_user_message.text}"  # noqa: E501
                )
                return identified_language
            except Exception:
                self.logger.info("Writing report in English")
                return "English"

    async def _generate_final_report(self) -> None:
        if self.research_topic is None:
            raise ValueError("No research topic set!")

        if self.research_plan is None:
            raise ValueError("No research plan set!")

        if len(self.interim_reports) == 0:
            raise ValueError("No interim reports available!")

        # await self._emit(TrajectoryEvent(title="Generating final report"))
        language = await self._get_language()

        prompt = ResearchPrompts.final_report_prompt(
            topic=self.research_topic,
            context=self._context,
            findings=self.interim_reports,
            language=language,
        )
        response: list[str] = []

        if settings.STREAMING is True:
            # Final report is streamed
            async with chat_pool.throttle():
                async for event, _ in self.chat_model.create(
                    messages=[UserMessage(content=prompt)], stream=True, max_retries=settings.MAX_RETRIES
                ):
                    if isinstance(event, ChatModelNewTokenEvent):
                        content = event.value.get_text_content()
                        response.append(content)
                        await self._emit(TextEvent(text=content))
                    elif isinstance(event, ChatModelSuccessEvent):
                        await self._emit(PassThroughEvent(event=event))
        else:
            output = await self.chat_model.create(
                messages=[UserMessage(content=prompt)], max_retries=settings.MAX_RETRIES
            )
            response.append(output.get_text_content())
            await self._emit(TextEvent(text="".join(response)))
            await self._emit(PassThroughEvent(event=ChatModelSuccessEvent(value=output)))

        self.final_report = "".join(response)

    async def _generate_research_plan(self) -> list[ResearchQuery]:
        if self.research_topic is None:
            raise ValueError("Research topic has not been set!")

        prompt = ResearchPrompts.research_plan_prompt(
            topic=self.research_topic, context=self._context, max_queries=settings.RESEARCH_PLAN_BREADTH
        )

        async with chat_pool.throttle():
            response = await self.structured_chat_model.create_structure(
                schema=ResearchPlanSchema,
                messages=[UserMessage(content=prompt)],
                max_retries=settings.MAX_RETRIES,
            )

        try:
            plan = ResearchPlanSchema(**response.object)
            return plan.questions
        except ValidationError as e:
            raise ValueError("Failed to generate a valid research plan!") from e

    async def _perform_research(self) -> None:
        if self.research_plan is None or len(self.research_plan) == 0:
            raise ValueError("No research plan has been set!")

        reports = await asyncio.gather(*(self._research_step(step) for step in self.research_plan))
        self.interim_reports.extend(reports)

    async def _research_step(self, query: ResearchQuery) -> ResearchReport:
        await self._emit(TrajectoryEvent(title="Researching", content=query.question))

        docs: list[Document] = await self.vector_store.asimilarity_search(
            query=" ".join([query.question, query.search_query]),
            k=settings.RESEARCH_MAX_DOCS_PER_STEP,
        )

        self.final_report_docs += docs

        self.logger.info(f"Generating research report {query.question}")

        research_report_prompt = ResearchPrompts.research_report_prompt(query=query, docs=docs)
        async with chat_pool.throttle():
            response = await self.chat_model.create(
                messages=[UserMessage(content=research_report_prompt)],
                max_retries=settings.MAX_RETRIES,
                max_tokens=settings.RESEARCH_FINDINGS_MAX_TOKENS,
            )
        report = response.get_text_content()
        return ResearchReport(query=query, report=report)

    async def _search_query(self, query: str, max_results: int = 3) -> list[SearchResult]:
        try:
            engine = SearchEngineFactory.create()
            # search engines do not throttle internally
            async with task_pool.throttle():
                return await engine.search(query=query, max_results=max_results)
        except Exception as e:
            self.logger.exception(repr(e))
        return []

    async def _generate_citations(self) -> None:
        if len(self.final_report_docs) > 0:
            # Compress docs
            await self._emit(GeneratingCitationsEvent())
            docs = self._dedup_documents_by_content(self.final_report_docs)
            generator = CitationGeneratorFactory.create()
            self.forward_events_from(generator)

            await generator.generate(docs=docs, response=self.final_report or "")
            await self._emit(GeneratingCitationsCompleteEvent())

    def _dedup_documents_by_content(self, documents: list[Document]) -> list[Document]:
        seen = set()
        deduped = []
        for doc in documents:
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                deduped.append(doc)
        return deduped
