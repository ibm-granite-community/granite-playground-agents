# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0


import pytest

from granite_core.chat_model import ChatModelFactory
from granite_core.search.engines.factory import SearchEngineFactory
from granite_core.search.filter import SearchResultsFilter
from granite_core.search.types import SearchResult


@pytest.mark.asyncio
async def test_basic_search() -> None:
    """Test basic search infrastructure"""
    engine = SearchEngineFactory.create()
    results = await engine.search(query="IBM", max_results=3)
    assert len(results) == 3
    assert results[0].title and results[0].body and results[0].url


@pytest.mark.asyncio
async def test_search_filter() -> None:
    """Test search filter"""
    chat_model = ChatModelFactory.create()
    filter = SearchResultsFilter(chat_model=chat_model)
    filtered_results = await filter.filter(
        "When was IBM founded?",
        [
            SearchResult(
                title="Gardening digest",
                body="The following gardening strategies boost plant nutrition, ensure healthy growing habits, deter pests, and have numerous other beneficial effects in gardens of various sizes.",  # noqa: E501
                href="https://www.gardeningdigest.com/",
            ),
            SearchResult(
                title="IBM Wikipedia Article",
                body="International Business Machines Corporation (using the trademark IBM), nicknamed Big Blue, is an American multinational technology company headquartered in Armonk, New York, and present in over 175 countries.",  # noqa: E501
                href="https://en.wikipedia.org/wiki/IBM",
            ),
        ],
    )

    assert len(filtered_results) == 1
    assert filtered_results[0].href == "https://en.wikipedia.org/wiki/IBM"
