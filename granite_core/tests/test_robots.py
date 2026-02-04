# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0


import asyncio

import httpx
import pytest

import granite_core.search.robots as robots
from granite_core.search.user_agent import UserAgent


@pytest.mark.asyncio
async def test_robots_allowed() -> None:
    """Test basic chat infrastructure"""
    async with httpx.AsyncClient(timeout=5) as client:
        assert await robots.can_fetch(client=client, url="https://www.ibm.com/about", user_agent=UserAgent().user_agent)
        assert await robots.can_fetch(client=client, url="https://www.wikipedia.com", user_agent=UserAgent().user_agent)
        assert await robots.can_fetch(client=client, url="https://www.netwes.com/", user_agent=UserAgent().user_agent)
        assert await robots.can_fetch(
            client=client,
            url="https://dataplatform.cloud.ibm.com/docs/content/wsj/getting-started/whats-new-wx.html?context=wx",
            user_agent=UserAgent().user_agent,
        )


@pytest.mark.asyncio
async def test_robots_forbidden() -> None:
    """Test basic chat infrastructure"""
    async with httpx.AsyncClient(timeout=5) as client:
        assert not await robots.can_fetch(client=client, url="https://facebook.com", user_agent=UserAgent().user_agent)
        assert not await robots.can_fetch(client=client, url="https://instagram.com", user_agent=UserAgent().user_agent)
        assert not await robots.can_fetch(
            client=client,
            url="https://www.reddit.com/r/IBM/comments/1dpl799/your_opinionview_on_granite_models/",
            user_agent=UserAgent().user_agent,
        )


@pytest.mark.asyncio
async def test_cache_ttl() -> None:
    """Test basic chat infrastructure"""
    async with httpx.AsyncClient(timeout=5) as client:
        default_cache_ttl = robots.CACHE_TTL

        # Reduce cache ttl to test invalidation
        robots.CACHE_TTL = 1

        old_robots_parser: robots.MutableRobotFileParser = await robots.get_robots_parser(
            client=client, robots_url="https://www.ibm.com/robots.txt"
        )

        await asyncio.sleep(robots.CACHE_TTL + 1)

        new_robots_parser: robots.MutableRobotFileParser = await robots.get_robots_parser(
            client=client, robots_url="https://www.ibm.com/robots.txt"
        )

        # The parser should be new
        assert old_robots_parser != new_robots_parser

        robots.CACHE_TTL = default_cache_ttl

        # Should cache hit
        cached_robots_parser: robots.MutableRobotFileParser = await robots.get_robots_parser(
            client=client, robots_url="https://www.ibm.com/robots.txt"
        )

        # Verify cache hit
        assert cached_robots_parser == new_robots_parser
