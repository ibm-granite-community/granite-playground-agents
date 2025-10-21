# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0


from typing import Literal

from beeai_framework.backend.types import ChatModelUsage
from pydantic import BaseModel


class UsageInfo(BaseModel):
    completion_tokens: int | None
    prompt_tokens: int | None
    total_tokens: int | None
    model_id: str
    type: Literal["usage_info"] = "usage_info"


def create_usage_info(
    usage: ChatModelUsage | None,
    model_id: str,
) -> UsageInfo:
    return UsageInfo(
        completion_tokens=usage.completion_tokens if usage else None,
        prompt_tokens=usage.prompt_tokens if usage else None,
        total_tokens=usage.total_tokens if usage else None,
        model_id=model_id,
    )
