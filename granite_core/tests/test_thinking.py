# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0


import pytest

from granite_core.thinking.response_parser import ThinkingResponseParser
from granite_core.thinking.stream_handler import TagStartEvent, ThinkingStreamHandler, TokenEvent


@pytest.mark.asyncio
async def test_streaming() -> None:
    """Test thinking stream"""

    thinking_text = "The user asked for me to think. I should think. I will also include a <tag></tag>"
    response_text = "I am done thinking!"

    stream = f"<think>{thinking_text}</think><response>{response_text}</response>"
    tokens = [stream[i : i + 4] for i in range(0, len(stream), 4)]
    handler = ThinkingStreamHandler(tags=["think", "response"])

    thinking: list[str] = []
    response: list[str] = []

    for token in tokens:
        for output in handler.on_token(token=token):
            if isinstance(output, TokenEvent):
                if output.tag == "think" and output.token:
                    thinking.append(output.token)
                elif output.tag == "response" and output.token:
                    response.append(output.token)
            elif isinstance(output, TagStartEvent):
                pass

    assert "".join(thinking) == thinking_text
    assert "".join(response) == response_text


@pytest.mark.asyncio
async def test_weird_streaming() -> None:
    """Test thinking stream with noise"""

    thinking_text = "The user asked for me to think. I should think. I will also include a <tag></tag>"
    response_text = "<response>I am done thinking!"

    stream = f"Random text. <think>{thinking_text}</think><response>{response_text}</response> Random text."
    tokens = [stream[i : i + 4] for i in range(0, len(stream), 4)]
    handler = ThinkingStreamHandler(tags=["think", "response"])

    thinking: list[str] = []
    response: list[str] = []

    for token in tokens:
        for output in handler.on_token(token=token):
            if isinstance(output, TokenEvent):
                if output.tag == "think" and output.token:
                    thinking.append(output.token)
                elif output.tag == "response" and output.token:
                    response.append(output.token)
            elif isinstance(output, TagStartEvent):
                pass

    assert "".join(thinking) == thinking_text
    assert "".join(response) == response_text


@pytest.mark.asyncio
async def test_parsing() -> None:
    """Test thinking parse"""
    thinking_text = "The user asked for me to think. I should think. I will also include a <tag></tag>"
    response_text = "I am done thinking!"
    stream = f"<think>{thinking_text}</think><response>{response_text}</response>"

    parser = ThinkingResponseParser()
    response = parser.parse(stream)

    assert response.thinking == thinking_text
    assert response.response == response_text
