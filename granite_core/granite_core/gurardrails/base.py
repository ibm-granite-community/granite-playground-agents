# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0


from abc import ABC, abstractmethod

from beeai_framework.backend import AnyMessage
from pydantic import BaseModel


class GuardrailResult(BaseModel):
    violated: bool
    reason: str | None = None


class Guardrail(ABC):
    @abstractmethod
    async def evaluate(self, messages: list[AnyMessage]) -> GuardrailResult:
        pass
