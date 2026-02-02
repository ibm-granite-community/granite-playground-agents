# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0


import httpx
import pytest

from granite_core.search.robots import can_fetch
from granite_core.search.user_agent import UserAgent


@pytest.mark.asyncio
async def test_robots_allowed() -> None:
    print("IMPORT TIME PRINT")
    """Test basic chat infrastructure"""
    async with httpx.AsyncClient(timeout=5.0) as client:
        assert await can_fetch(client=client, url="https://www.ibm.com/about", user_agent=UserAgent().user_agent)
        assert await can_fetch(client=client, url="https://www.wikipedia.com", user_agent=UserAgent().user_agent)


@pytest.mark.asyncio
async def test_robots_forbidden() -> None:
    """Test basic chat infrastructure"""
    async with httpx.AsyncClient(timeout=5.0) as client:
        assert not await can_fetch(client=client, url="https://facebook.com", user_agent=UserAgent().user_agent)
        assert not await can_fetch(client=client, url="https://instagram.com", user_agent=UserAgent().user_agent)
