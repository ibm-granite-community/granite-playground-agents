# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0


import pytest

from granite_core.search.robots import can_fetch
from granite_core.search.user_agent import UserAgent


@pytest.mark.asyncio
async def test_robots_allowed() -> None:
    """Test basic chat infrastructure"""

    assert await can_fetch(url="https://www.ibm.com/about", user_agent=UserAgent().user_agent)


@pytest.mark.asyncio
async def test_robots_forbidden() -> None:
    """Test basic chat infrastructure"""

    assert not await can_fetch(url="https://facebook.com", user_agent=UserAgent().user_agent)
