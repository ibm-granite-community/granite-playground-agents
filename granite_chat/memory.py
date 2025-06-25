from acp_sdk import MessagePart
from beeai_framework.backend import Message

from granite_chat.config import settings  # type: ignore


def estimate_tokens(messages: list[Message]) -> int:
    token_estimate = 0

    for msg in messages:
        token_estimate += len(msg.text) // 4

    return token_estimate


def exceeds_token_limit(messages: list[Message]) -> bool:
    token_count = estimate_tokens(messages)
    return token_count >= settings.CHAT_TOKEN_LIMIT


def token_limit_message_part() -> MessagePart:
    return MessagePart(
        content_type="limit",
        content="Your message will exceed the length limit for this chat.",
        role="system",
    )  # type: ignore[call-arg]
