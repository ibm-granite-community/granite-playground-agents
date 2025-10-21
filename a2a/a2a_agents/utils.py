# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0


from a2a.types import Message as A2AMessage
from a2a.types import Role
from beeai_framework.backend import AssistantMessage, UserMessage
from beeai_framework.backend.message import AnyMessage
from granite_core.logging import get_logger

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
