# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0

import ast
import json
from typing import Any

from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain.messages import ToolCall
from langgraph.graph.message import REMOVE_ALL_MESSAGES, RemoveMessage
from langgraph.runtime import Runtime


class PatchInvalidToolCallsMiddleware(AgentMiddleware):
    def after_model(self, state: AgentState, runtime: Runtime[Any]) -> dict[str, Any] | None:
        """After the agent runs, handle dangling tool calls from any AIMessage."""
        messages = state["messages"]
        if not messages or len(messages) == 0:
            return None

        patched_messages = []

        for _, msg in enumerate(messages):
            patched_messages.append(msg)
            if msg.type == "ai" and msg.invalid_tool_calls:
                for tool_call in msg.invalid_tool_calls:
                    if tool_call["name"] and tool_call["id"] and tool_call["args"] and tool_call["error"] is None:
                        args = str_to_dict(tool_call["args"] or "")
                        msg.tool_calls.append(ToolCall(name=tool_call["name"], id=tool_call["id"], args=args))
        return {
            "messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), *patched_messages],
        }


def str_to_dict(s: str) -> dict:
    """
    Convert a string to a dictionary.
    Handles:
    - JSON strings
    - Python-style dict strings
    - Double-encoded JSON strings
    Raises ValueError if conversion fails.
    """
    if not isinstance(s, str):
        raise ValueError("Input must be a string")

    s = s.strip()  # remove whitespace

    # Helper to attempt json.loads safely
    def try_json_load(x: str) -> Any:
        try:
            return json.loads(x)
        except (json.JSONDecodeError, TypeError):
            return None

    # Step 1: Try JSON
    d = try_json_load(s)
    if isinstance(d, dict):
        return d

    # Step 2: Handle double-encoded JSON (string inside JSON)
    if isinstance(d, str):
        inner = try_json_load(d)
        if isinstance(inner, dict):
            return inner

    # Step 3: Fallback to Python-style dict
    try:
        d = ast.literal_eval(s)
        if isinstance(d, dict):
            return d
    except (ValueError, SyntaxError):
        pass

    raise ValueError("String could not be converted to a dictionary")
