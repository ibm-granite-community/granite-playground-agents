# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0


import json
from typing import Literal

from granite_core import utils
from granite_core.config import settings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_ibm import ChatWatsonx
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from pydantic import SecretStr


class ChatModelFactory:
    """Factory for ChatModel instances."""

    @staticmethod
    def create(streaming: bool = False) -> BaseChatModel:
        provider: Literal["openai", "watsonx", "ollama"] = settings.LLM_PROVIDER
        model_id: str = settings.LLM_MODEL

        max_tokens: int = settings.MAX_TOKENS
        temperature: float = settings.TEMPERATURE

        if provider == "openai":
            base_url: str = str(settings.LLM_API_BASE)
            api_key: str | None = utils.get_secret_value(setting=settings.LLM_API_KEY)

            if api_key is None:
                raise ValueError("LLM_API_KEY is not set")

            return ChatOpenAI(
                model=model_id,
                stream_usage=streaming,
                temperature=temperature,
                max_completion_tokens=max_tokens,
                api_key=SecretStr(secret_value=api_key),
                base_url=base_url + "/v1",
                default_headers=(
                    json.loads(settings.LLM_API_HEADERS.get_secret_value()) if settings.LLM_API_HEADERS else {}
                ),
            )
        elif provider == "watsonx":
            base_url = str(settings.WATSONX_API_BASE)
            api_key = utils.get_secret_value(setting=settings.WATSONX_API_KEY)
            project_id: str | None = utils.get_secret_value(setting=settings.WATSONX_PROJECT_ID)

            if api_key is None:
                raise ValueError("LLM_API_KEY is not set")

            return ChatWatsonx(
                model_id=model_id,
                api_key=SecretStr(secret_value=api_key),
                url=SecretStr(secret_value=base_url),
                project_id=project_id,
                streaming=streaming,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        elif provider == "ollama":
            base_url = str(settings.OLLAMA_BASE_URL) if settings.OLLAMA_BASE_URL else None  # type: ignore[assignment]

            # BeeAI Framework doesn't check if the URL ends with a slash (and assumes it doesn't)
            if base_url is not None and base_url.endswith("/"):
                base_url = base_url[:-1]

            return ChatOllama(
                model=model_id,
                base_url=base_url,
                temperature=temperature,
                num_predict=max_tokens,
            )
