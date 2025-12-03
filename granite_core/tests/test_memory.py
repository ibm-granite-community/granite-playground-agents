# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0


import pytest
from beeai_framework.backend import AssistantMessage, Message, UserMessage

from granite_core.memory import estimate_tokens


@pytest.mark.asyncio
async def test_token_estimate() -> None:
    """Test token count estimation"""

    messages: list[Message] = [UserMessage("Hello!"), AssistantMessage("Hello there! How can I assist you?")]

    assert estimate_tokens(messages) == 15
