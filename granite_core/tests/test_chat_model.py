# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch

import pytest
from beeai_framework.adapters.ollama import OllamaChatModel
from beeai_framework.adapters.openai import OpenAIChatModel
from beeai_framework.adapters.watsonx import WatsonxChatModel
from pydantic.networks import HttpUrl

from granite_core.chat_model import ChatModelFactory
from granite_core.config import Settings


@patch("granite_core.chat_model.settings")
def test_create_openai_chat_model(mock_settings: Settings) -> None:
    mock_settings.LLM_PROVIDER = "openai"
    chat_model = ChatModelFactory.create()
    assert isinstance(chat_model, OpenAIChatModel)


@patch("granite_core.chat_model.settings")
def test_create_watsonx_chat_model(mock_settings: Settings) -> None:
    mock_settings.LLM_PROVIDER = "watsonx"
    chat_model = ChatModelFactory.create()
    assert isinstance(chat_model, WatsonxChatModel)


@patch("granite_core.chat_model.settings")
def test_create_ollama_chat_model(mock_settings: Settings) -> None:
    mock_settings.LLM_PROVIDER = "ollama"
    mock_settings.OLLAMA_BASE_URL = HttpUrl("http://localhost:11434/")
    chat_model = ChatModelFactory.create(model_type="structured")
    assert isinstance(chat_model, OllamaChatModel)


@patch("granite_core.chat_model.settings")
def test_create_bad_chat_model(mock_settings: Settings) -> None:
    mock_settings.LLM_PROVIDER = "fake"  # type: ignore[assignment]
    with pytest.raises(ValueError):
        ChatModelFactory.create()
