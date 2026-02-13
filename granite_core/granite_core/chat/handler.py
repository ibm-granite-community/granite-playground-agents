# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0


import asyncio
from collections.abc import Sequence
from typing import Any

from beeai_framework.backend import (
    AnyMessage,
    ChatModel,
    ChatModelNewTokenEvent,
    ChatModelSuccessEvent,
    SystemMessage,
)
from beeai_framework.backend.types import ChatModelOutput

from granite_core.chat.prompts import ChatPrompts
from granite_core.emitter import EventEmitter
from granite_core.events import PassThroughEvent, TextEvent, TokenLimitExceededEvent
from granite_core.gurardrails.base import Guardrail, GuardrailResult
from granite_core.gurardrails.copyright import CopyrightViolationGuardrail
from granite_core.gurardrails.web_access import WebAccessGuardrail
from granite_core.logging import get_logger_with_prefix
from granite_core.memory import estimate_tokens
from granite_core.work import chat_pool


class ChatHandler(EventEmitter):
    """
    Handler for basic chat interactions.

    Receives messages and emits response tokens through the event system.
    Includes guardrails, capability checking and streaming support.
    """

    def __init__(
        self,
        chat_model: ChatModel,
        session_id: str | None = None,
        token_limit: int | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the ChatHandler.

        Args:
            chat_model: The chat model to use for generating responses
            session_id: Optional session identifier for logging
            token_limit: Optional token limit. If None, uses CHAT_TOKEN_LIMIT from settings.
        """
        super().__init__(*args, **kwargs)
        self.chat_model = chat_model
        self.session_id = session_id or "default"
        self.token_limit = token_limit
        self.logger = get_logger_with_prefix(logger_name=__name__, tool_name="ChatHandler", session_id=self.session_id)

        # copyright and web access guardrails
        self.guardrails: Sequence[Guardrail] = [
            CopyrightViolationGuardrail(chat_model=self.chat_model),
            WebAccessGuardrail(chat_model=self.chat_model),
        ]

        self.logger.debug(msg="Initialized ChatHandler")

    async def run(
        self,
        messages: list[AnyMessage],
        stream: bool = True,
    ) -> None:
        """
        Process messages and generate a response.

        Emits events for the response. Callers should subscribe to events to receive the output.

        Args:
            messages: Sequence of messages to process (conversation history)
            stream: Whether to stream the response (emits TextEvent for each token)
        """
        self.logger.info(f"Processing chat with {len(messages)} messages")

        # Evaluate all guardrails concurrently
        guardrail_results: list[GuardrailResult] = await asyncio.gather(
            *[guardrail.evaluate(messages=messages) for guardrail in self.guardrails]
        )

        # Map results back to guardrails for precedence checking
        results_with_guardrails = list(zip(self.guardrails, guardrail_results, strict=True))

        # Check for copyright violation first (highest precedence)
        copyright_violation = None
        web_access_violation = None

        for guardrail, result in results_with_guardrails:
            if result.violated:
                if isinstance(guardrail, CopyrightViolationGuardrail):
                    copyright_violation = result
                    self.logger.warning(msg=f"Copyright guardrail violated: {result.reason}")
                elif isinstance(guardrail, WebAccessGuardrail):
                    web_access_violation = result
                    self.logger.warning(msg=f"Web access guardrail violated: {result.reason}")

        # Prepare system message based on precedence
        if copyright_violation:
            # Copyright violation has highest precedence
            system_message = SystemMessage(
                content=f"Providing an answer to the user would result in a potential copyright violation.\n"
                f"Reason: {copyright_violation.reason}\n\n"
                f"Inform the user and suggest alternatives."
            )
        elif web_access_violation:
            # Web access violation is checked if copyright passes
            system_message = SystemMessage(
                content=f"You cannot answer this request because it requires web search or internet access.\n"
                f"Reason: {web_access_violation.reason}\n\n"
            )
        else:
            # No violations, use standard system prompt
            system_message = SystemMessage(content=ChatPrompts.chat_system_prompt())

        # Prepend system message to conversation
        full_messages: list[Any] = [system_message, *messages]

        # Check token limit
        if self.token_limit is not None:
            estimated_tokens = estimate_tokens(full_messages)
            if estimated_tokens >= self.token_limit:
                self.logger.warning(f"Token limit exceeded: {estimated_tokens} > {self.token_limit}")
                await self._emit(
                    TokenLimitExceededEvent(estimated_tokens=estimated_tokens, token_limit=self.token_limit)
                )
                return

        # Generate response
        response_text_parts: list[str] = []

        async with chat_pool.throttle():
            if stream:
                # Stream response and emit events
                async for event, _ in self.chat_model.run(full_messages, stream=True):
                    if isinstance(event, ChatModelNewTokenEvent):
                        token: str = event.value.get_text_content()
                        response_text_parts.append(token)
                        # Emit text event for streaming
                        await self._emit(event=TextEvent(text=token))
                    elif isinstance(event, ChatModelSuccessEvent):
                        # Emit success event when streaming
                        await self._emit(event=PassThroughEvent(event=event))
            else:
                # Non-streaming response
                output: ChatModelOutput = await self.chat_model.run(full_messages, stream=False)
                response_text: str = output.get_text_content()
                response_text_parts.append(response_text)

                # Emit full text for non-streaming
                full_response: str = "".join(response_text_parts)
                await self._emit(event=TextEvent(text=full_response))
                # Emit success event
                await self._emit(event=PassThroughEvent(event=ChatModelSuccessEvent(value=output)))
