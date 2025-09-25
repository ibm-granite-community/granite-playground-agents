import math
from typing import Literal

from beeai_framework.backend import Message
from pydantic import BaseModel

from granite_core.config import settings


class TokenLimitExceeded(BaseModel):
    type: Literal["limit"] = "limit"
    code: Literal["token_limit_exceeded"] = "token_limit_exceeded"
    estimated_tokens_used: int


def estimate_tokens(messages: list[Message]) -> int:
    """
    Estimate tokens by counting characters and divide by 4
    """
    token_estimate = 0

    for msg in messages:
        token_estimate += math.ceil(len(msg.text) / 4)
        token_estimate += math.ceil(len(msg.role) / 4)

    return token_estimate


def exceeds_token_limit(token_count: int) -> bool:
    return token_count >= settings.CHAT_TOKEN_LIMIT


def token_limit_response(tokens_used: int) -> BaseModel:
    return TokenLimitExceeded(estimated_tokens_used=tokens_used)
