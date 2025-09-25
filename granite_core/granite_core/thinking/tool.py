from typing import Any

from beeai_framework.backend import (
    ChatModel,
    ChatModelNewTokenEvent,
    ChatModelSuccessEvent,
    Message,
    SystemMessage,
)

from granite_core import get_logger_with_prefix
from granite_core.config import settings
from granite_core.emitter import EventEmitter
from granite_core.events import (
    PassThroughEvent,
    TextEvent,
    ThinkEvent,
)
from granite_core.thinking.prompts import ThinkingPrompts
from granite_core.work import chat_pool


class ThinkingTool(EventEmitter):
    def __init__(
        self,
        chat_model: ChatModel,
        messages: list[Message],
        session_id: str,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.chat_model = chat_model
        self.messages = messages
        self.session_id = session_id
        self.logger = get_logger_with_prefix(__name__, tool_name="Researcher", session_id=session_id)

    async def run(self) -> None:
        thinking_tokens: list[str] = []

        async with chat_pool.throttle():
            async for event, _ in self.chat_model.create(
                messages=[
                    SystemMessage(content=ThinkingPrompts.two_step_thinking_system_prompt()),
                    *self.messages,
                ],
                stream=True,
                max_retries=settings.MAX_RETRIES,
            ):
                if isinstance(event, ChatModelNewTokenEvent):
                    content = event.value.get_text_content()
                    thinking_tokens.append(content)
                    await self._emit(ThinkEvent(text=content))

        thinking_str = "".join(thinking_tokens)

        async with chat_pool.throttle():
            async for event, _ in self.chat_model.create(
                messages=[
                    SystemMessage(
                        content=ThinkingPrompts.two_step_thinking_answer_system_prompt(thinking=thinking_str)
                    ),
                    *self.messages,
                ],
                stream=True,
                max_retries=settings.MAX_RETRIES,
            ):
                if isinstance(event, ChatModelNewTokenEvent):
                    content = event.value.get_text_content()
                    await self._emit(TextEvent(text=content))
                elif isinstance(event, ChatModelSuccessEvent):
                    await self._emit(PassThroughEvent(event=event))
