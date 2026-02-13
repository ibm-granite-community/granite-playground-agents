# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0


from beeai_framework.backend import AnyMessage, ChatModel, SystemMessage
from pydantic import BaseModel, Field

from granite_core.config import settings
from granite_core.gurardrails.base import Guardrail, GuardrailResult
from granite_core.logging import get_logger
from granite_core.work import chat_pool

logger = get_logger(__name__)


class WebAccessRequirementSchema(BaseModel):
    requires_web_access: bool = Field(
        description="Indicate if the user's request requires web search or internet access to answer properly"
    )
    reason: str = Field(description="Brief reasoning about why web access is or isn't needed")


class WebAccessGuardrail(Guardrail):
    """
    Guardrail that detects when a user request requires web search or internet access.

    This is useful for chat handlers that don't have web access capabilities,
    allowing them to inform users that they need to use a different agent/tool.
    """

    def __init__(self, chat_model: ChatModel) -> None:
        super().__init__()
        self.chat_model = chat_model

    def system_prompt(self) -> str:
        return """
Evaluate if the user's request requires web search or internet access to answer properly.

Criteria for requiring web access:
- Requests for current/recent information (news, stock prices, weather, sports scores, etc.)
- Requests for real-time data or live information
- Requests to search for or find specific websites, articles, or online resources
- Requests for information about recent events (within the last few months)
- Requests that explicitly mention "search", "look up", "find online", etc.
- Requests for information that changes frequently (exchange rates, trending topics, etc.)

Does NOT require web access:
- General knowledge questions about established facts, history, science, etc.
- Requests for explanations, summaries, or analysis of provided information
- Creative tasks (writing, brainstorming, coding help, etc.)
- Questions about well-known concepts, theories, or historical events
- Math, logic, or reasoning problems
- Requests for advice or opinions based on general knowledge

When in doubt, if the information could reasonably be in the model's training data and doesn't need to be current, mark as NOT requiring web access.
"""  # noqa: E501

    async def evaluate(self, messages: list[AnyMessage]) -> GuardrailResult:
        logger.info("Evaluating messages for web access requirement")
        async with chat_pool.throttle():
            response = await self.chat_model.run(
                [SystemMessage(self.system_prompt()), *messages],
                response_format=WebAccessRequirementSchema,
                max_retries=settings.MAX_RETRIES,
            )

        assert isinstance(response.output_structured, WebAccessRequirementSchema)
        guardrail = response.output_structured

        if guardrail.requires_web_access:
            logger.warning(f"Web access required: {guardrail.reason}")
        else:
            logger.info(f"No web access required: {guardrail.reason}")

        return GuardrailResult(violated=guardrail.requires_web_access, reason=guardrail.reason)
