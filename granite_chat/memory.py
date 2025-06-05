from beeai_framework.backend import Message


def estimate_tokens(messages: list[Message]) -> int:
    token_estimate = 0

    for msg in messages:
        token_estimate += len(msg.text) // 4

    return token_estimate
