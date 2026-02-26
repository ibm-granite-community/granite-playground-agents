# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0


import pytest
from beeai_framework.backend import AnyMessage, AssistantMessage, UserMessage

from granite_core.chat_model import ChatModelFactory
from granite_core.citations.events import CitationEvent
from granite_core.citations.types import Citation
from granite_core.emitter import Event
from granite_core.events import GeneratingCitationsCompleteEvent, GeneratingCitationsEvent, TextEvent, TrajectoryEvent
from granite_core.research.prompts import ResearchPrompts
from granite_core.research.researcher import Researcher
from granite_core.research.types import ResearchQuery


@pytest.mark.asyncio
async def test_basic_researcher() -> None:
    """Test basic research infrastructure"""
    chat_model = ChatModelFactory.create()
    structured_chat_model = ChatModelFactory.create(model_type="structured")
    messages: list[AnyMessage] = [
        UserMessage("Hello!"),
        AssistantMessage("Hello! How can I help you today?"),
        UserMessage("Write a brief report on Geoffrey Hinton and his contributions to AI."),
    ]

    researcher = Researcher(
        chat_model=chat_model,
        structured_chat_model=structured_chat_model,
        messages=messages,
        session_id="test_session",
    )

    final_agent_response_text: list[str] = []
    final_citations: list[Citation] = []

    async def research_listener(event: Event) -> None:
        if isinstance(event, TextEvent):
            final_agent_response_text.append(event.text)
        elif isinstance(event, TrajectoryEvent):
            assert len(event.title) > 0
        elif isinstance(event, GeneratingCitationsEvent):
            assert True
        elif isinstance(event, CitationEvent):
            assert isinstance(event.citation, Citation)
            final_citations.append(event.citation)
        elif isinstance(event, GeneratingCitationsCompleteEvent):
            assert True

    researcher.subscribe(handler=research_listener)
    await researcher.run()
    assert len(final_agent_response_text) > 0
    assert len(final_citations) > 0
    assert "Geoffrey Hinton" in "".join(final_agent_response_text)


@pytest.mark.asyncio
async def test_interactive_researcher() -> None:
    """Test basic research infrastructure"""
    chat_model = ChatModelFactory.create()
    structured_chat_model = ChatModelFactory.create(model_type="structured")
    messages: list[AnyMessage] = [
        UserMessage("In interested in binary star systems!"),
    ]

    researcher = Researcher(
        chat_model=chat_model,
        structured_chat_model=structured_chat_model,
        messages=messages,
        session_id="test_session",
        interactive=True,
    )

    final_agent_response_text: list[str] = []

    async def research_listener(event: Event) -> None:
        if isinstance(event, TextEvent):
            final_agent_response_text.append(event.text)
        elif isinstance(
            event, (TrajectoryEvent, GeneratingCitationsEvent, CitationEvent, GeneratingCitationsCompleteEvent)
        ):
            # Agent should not execute research flow
            raise AssertionError()

    researcher.subscribe(handler=research_listener)
    await researcher.run()
    # Agent should ask user for clarification
    assert "?" in "".join(final_agent_response_text)


def test_research_prompts() -> None:
    """
    Test research prompts
    """
    # note: this function is not called anywhere so testing directly for coverage in case it's used in the future

    query = ResearchQuery(
        question="What are the latest advancements in AI?",
        search_query="latest advancements in AI",
        rationale="To understand the current state of AI technology.",
    )
    context = "AI has seen rapid advancements in recent years."
    max_queries = 5
    prompt = ResearchPrompts.generate_search_queries_prompt(query, context, max_queries)
    assert query.question in prompt
    assert query.search_query in prompt
    assert context in prompt
    assert f"Generate {max_queries}" in prompt
