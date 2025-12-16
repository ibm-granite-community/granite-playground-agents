# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0


from typing import Annotated

from a2a.types import Message as A2AMessage
from a2a.types import Role
from agentstack_sdk.a2a.extensions import (
    EmbeddingServiceExtensionServer,
    EmbeddingServiceExtensionSpec,
    LLMServiceExtensionServer,
    LLMServiceExtensionSpec,
)
from beeai_framework.backend import AssistantMessage, UserMessage
from beeai_framework.backend.message import AnyMessage
from granite_core.config import settings as core_settings
from granite_core.logging import get_logger
from openai import AsyncOpenAI
from pydantic import SecretStr

from a2a_agents.config import settings

logger = get_logger(__name__)


def to_framework_messages(history: list[A2AMessage]) -> list[AnyMessage]:
    """
    Converts a list of messages into a list of framework messages, separating user and agent turns.

    Args:
        history (list[Message]): A list of messages containing user and agent turns.

    Returns:
        list[AnyMessage]: A list of framework messages, where each message represents
            either a user or an agent's input, depending on the turn.
    """

    # hold all text from this turn e.g. User or Agent
    text_from_this_turn = ""
    # mark whether User or Agent turns are being processed, user always goes first
    current_role = Role.user
    # return a list of framework messages
    framework_messages: list[AnyMessage] = []

    for message in history:
        # there can be multiple message parts in any given message, join these together
        all_parts = "".join(part.root.text for part in message.parts if part.root.kind == "text")

        # test whether there has been a change between User or Agent messages
        if message.role == current_role:
            # if this message has the same Role as the previous message, buffer the joined message parts
            text_from_this_turn += all_parts
        else:
            # if this message does not have the same Role as the previous message
            match message.role:
                case Role.agent:
                    # if switching to Agent, add a User message to the output
                    framework_messages.append(UserMessage(text_from_this_turn))
                    current_role = Role.agent
                case Role.user:
                    # if switching to User, add an Agent message to the output
                    framework_messages.append(AssistantMessage(text_from_this_turn))
                    current_role = Role.user
                case _:
                    logger.error("Unknown user role in message history")

            # when switching between User and Agent, start a new buffer
            text_from_this_turn = all_parts

    return framework_messages


async def configure_models(
    llm_ext: Annotated[
        LLMServiceExtensionServer,
        LLMServiceExtensionSpec.single_demand(suggested=(settings.SUGGESTED_LLM_MODEL,)),
    ],
    embedding_ext: Annotated[
        EmbeddingServiceExtensionServer | None,
        EmbeddingServiceExtensionSpec.single_demand(suggested=(settings.SUGGETED_EMBEDDING_MODEL,)),
        None,
    ] = None,
) -> None:
    if settings.USE_AGENTSTACK_LLM and llm_ext and llm_ext.data and llm_ext.data.llm_fulfillments:
        [llm_config] = llm_ext.data.llm_fulfillments.values()

        core_settings.LLM_PROVIDER = "openai"
        core_settings.LLM_MODEL = llm_config.api_model
        core_settings.LLM_API_BASE = llm_config.api_base
        core_settings.LLM_API_KEY = SecretStr(llm_config.api_key)

        logger.info(f"Model: {core_settings.LLM_MODEL} at {core_settings.LLM_API_BASE}")

    if (
        settings.USE_AGENTSTACK_LLM
        and embedding_ext
        and embedding_ext.data
        and embedding_ext.data.embedding_fulfillments
    ):
        [embedding_config] = embedding_ext.data.embedding_fulfillments.values()

        core_settings.EMBEDDINGS_PROVIDER = "openai"
        core_settings.EMBEDDINGS_MODEL = embedding_config.api_model
        core_settings.EMBEDDINGS_OPENAI_API_BASE = embedding_config.api_base
        core_settings.EMBEDDINGS_OPENAI_API_KEY = SecretStr(embedding_config.api_key)

        # set the embedding dimension
        embedding_client = AsyncOpenAI(
            api_key=core_settings.EMBEDDINGS_OPENAI_API_KEY.get_secret_value(),
            base_url=str(core_settings.EMBEDDINGS_OPENAI_API_BASE),
        )
        embedding_result = await embedding_client.embeddings.create(
            input="Hello, world!",
            model=core_settings.EMBEDDINGS_MODEL,
            encoding_format="float",
        )
        core_settings.EMBEDDINGS_MAX_SEQUENCE = len(embedding_result.data[0].embedding)
        core_settings.EMBEDDINGS_SIM_MAX_SEQUENCE = core_settings.EMBEDDINGS_MAX_SEQUENCE

        # set to None because we can't dynamically match to the user's chosen embedding model
        core_settings.EMBEDDINGS_HF_TOKENIZER = None

        logger.info(
            f"Embedding Model: {core_settings.EMBEDDINGS_MODEL} with length {core_settings.EMBEDDINGS_MAX_SEQUENCE} at {core_settings.LLM_API_BASE}"  # noqa: E501
        )
