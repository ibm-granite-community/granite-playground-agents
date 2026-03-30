# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0

import ast
import json
from collections.abc import Awaitable, Callable
from typing import Any

from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain.agents.middleware.types import ModelCallResult
from langchain.messages import AIMessage, ToolCall
from langchain_core.messages.base import BaseMessage


class PatchInvalidToolCallsMiddleware(AgentMiddleware):
    # Fixes broken granite tool calling
    # See https://github.com/langchain-ai/langchain-ibm/pull/117
    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        response: ModelResponse = await handler(request)
        ai_msg: BaseMessage = response.result[0]

        if isinstance(ai_msg, AIMessage):
            fixed_tool_call_ids: list[Any] = []
            for tool_call in ai_msg.invalid_tool_calls:
                if tool_call["name"] and tool_call["args"]:
                    args: str = tool_call["args"]
                    if args.startswith('"') and args.endswith('"'):
                        args = tool_call["args"][1:-1].encode().decode(encoding="unicode_escape")
                    try:
                        parsed = json.loads(s=args)
                    except json.JSONDecodeError:
                        parsed = ast.literal_eval(node_or_string=args)

                    ai_msg.tool_calls.append(ToolCall(name=tool_call["name"], args=parsed, id=tool_call["id"]))
                    fixed_tool_call_ids.append(tool_call["id"])

            # Remove only the invalid_tool_calls that were successfully fixed
            ai_msg.invalid_tool_calls = [tc for tc in ai_msg.invalid_tool_calls if tc["id"] not in fixed_tool_call_ids]

        return ModelResponse(
            result=[ai_msg],
            structured_response=response.structured_response,
        )
