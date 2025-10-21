# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0


import re

from pydantic import BaseModel


class ThinkingResponse(BaseModel):
    thinking: str | None = None
    response: str | None = None


class ThinkingResponseParser:
    def parse(self, text: str) -> ThinkingResponse:
        think_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
        response_match = re.search(r"<response>(.*?)</response>", text, re.DOTALL)

        think_content = think_match.group(1) if think_match else None
        response_content = response_match.group(1) if response_match else None

        return ThinkingResponse(thinking=think_content, response=response_content)
