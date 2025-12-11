# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0


from beeai_framework.backend import AnyMessage, ChatModel, SystemMessage
from pydantic import BaseModel, Field

from granite_core.config import settings
from granite_core.gurardrails.base import Guardrail, GuardrailResult
from granite_core.work import chat_pool


class CopyrightViolationSchema(BaseModel):
    is_copyright_violation: bool = Field(
        description="Indicate if the user message could lead to exposure of copyrighted Information"
    )
    reason: str = Field(description="Brief reasoning regarding the potential for copyright violation if present")


class CopyrightViolationGuardrail(Guardrail):
    def __init__(self, chat_model: ChatModel) -> None:
        super().__init__()
        self.chat_model = chat_model

    def system_prompt(self) -> str:
        return """
Evaluate the dialog for copyright risk. Pay close attention to the user's last message and decide if the assistant could potentially risk copyright law by answering.

Criteria:
When evaluating an input, treat it as a copyright risk if the user is requesting long or verbatim reproduction of copyrighted text they have not provided; requests for summaries, analyses, paraphrases, explanations, or short excerpts are allowed. Requests to output substantial portions of modern books, songs, poems, scripts, or other creative works should be flagged, unless the user includes the text themselves. In short: if the user wants you to produce copyrighted text you don’t already have in the prompt, it’s disallowed; if they want you to explain or transform provided text, it’s allowed.
"""  # noqa: E501, RUF001

    async def evaluate(self, messages: list[AnyMessage]) -> GuardrailResult:
        async with chat_pool.throttle():
            response = await self.chat_model.run(
                [SystemMessage(self.system_prompt()), *messages],
                response_format=CopyrightViolationSchema,
                max_retries=settings.MAX_RETRIES,
            )

        assert isinstance(response.output_structured, CopyrightViolationSchema)
        guardrail = response.output_structured

        return GuardrailResult(is_harmful=guardrail.is_copyright_violation, reason=guardrail.reason)
