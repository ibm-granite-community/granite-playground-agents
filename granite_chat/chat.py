from collections.abc import AsyncGenerator
from typing import Any

from beeai_framework.adapters.openai import OpenAIChatModel
from beeai_framework.adapters.watsonx import WatsonxChatModel
from beeai_framework.backend import (
    ChatModel,
    ChatModelNewTokenEvent,
    ChatModelParameters,
    ChatModelStructureOutput,
    ChatModelSuccessEvent,
)
from beeai_framework.backend.message import AnyMessage
from beeai_framework.backend.types import (
    ChatModelOutput,
)
from pydantic import BaseModel

from granite_chat.config import settings
from granite_chat.work import chat_pool


class ChatModelService:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._chat_model = ChatModelFactory.create()

    async def create(self, messages: list[AnyMessage]) -> ChatModelOutput:
        async with chat_pool.throttle():
            return await self._chat_model.create(messages=messages)

    async def create_stream(
        self, messages: list[AnyMessage]
    ) -> AsyncGenerator[ChatModelNewTokenEvent | ChatModelSuccessEvent, None]:
        async with chat_pool.throttle():
            async for data, event in self._chat_model.create(messages=messages, stream=True):
                match (data, event.name):
                    case (ChatModelNewTokenEvent(), "new_token"):
                        yield data
                    case (ChatModelSuccessEvent(), "success"):
                        yield data

    async def create_structure(self, schema: type[BaseModel], messages: list[AnyMessage]) -> ChatModelStructureOutput:
        async with chat_pool.throttle():
            return await self._chat_model.create_structure(schema=schema, messages=messages)

    @property
    def model_id(self) -> str:
        return self._chat_model.model_id


class ChatModelFactory:
    """Factory for ChatModel instances."""

    @staticmethod
    def create() -> ChatModel:
        provider = settings.LLM_PROVIDER
        model_id = settings.LLM_MODEL
        max_tokens = settings.MAX_TOKENS
        temperature = settings.TEMPERATURE

        if provider == "openai":
            base_url = str(settings.LLM_API_BASE)
            api_key = settings.LLM_API_KEY

            return OpenAIChatModel(
                model_id=model_id,
                api_key=api_key,
                base_url=base_url,
                parameters=ChatModelParameters(max_tokens=max_tokens, temperature=temperature),
            )
        elif provider == "watsonx":
            base_url = str(settings.WATSONX_API_BASE)
            api_key = settings.WATSONX_API_KEY
            project_id = settings.WATSONX_PROJECT_ID
            region = settings.WATSONX_REGION

            return WatsonxChatModel(
                model_id=model_id,
                api_key=api_key,
                base_url=base_url,
                project_id=project_id,
                region=region,
                parameters=ChatModelParameters(max_tokens=max_tokens, temperature=temperature),
            )
        else:
            raise ValueError("Unknown inference provider")
