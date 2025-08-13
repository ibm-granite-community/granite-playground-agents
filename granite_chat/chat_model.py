from beeai_framework.adapters.openai import OpenAIChatModel
from beeai_framework.adapters.watsonx import WatsonxChatModel
from beeai_framework.backend import (
    ChatModel,
    ChatModelParameters,
)

from granite_chat.config import settings


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
