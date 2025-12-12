# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0


from beeai_framework.backend import AssistantMessage, Message, UserMessage

from granite_core.config import settings
from granite_core.memory import TokenLimitExceeded, estimate_tokens, exceeds_token_limit, token_limit_response


def test_token_estimate() -> None:
    """Test token count estimation"""

    messages: list[Message] = [UserMessage("Hello!"), AssistantMessage("Hello there! How can I assist you?")]

    assert estimate_tokens(messages) == 15


def test_exceeds_token_limit() -> None:
    token_count = settings.CHAT_TOKEN_LIMIT
    assert exceeds_token_limit(token_count)
    assert not exceeds_token_limit(token_count - 1)


def test_token_limit_response() -> None:
    tokens_used = 15
    response = token_limit_response(tokens_used)
    assert isinstance(response, TokenLimitExceeded)
    assert response.type == "limit"
    assert response.code == "token_limit_exceeded"
    assert response.estimated_tokens_used == tokens_used
