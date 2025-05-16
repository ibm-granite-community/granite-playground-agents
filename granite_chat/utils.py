from acp_sdk import Message as ACPMessage
from beeai_framework.backend import UserMessage, AssistantMessage, Message as FrameworkMessage


def to_beeai_framework(messages: list[ACPMessage]):

    fw_messages: list[FrameworkMessage] = []

    for msg in messages:
        if "role" in msg.parts[0].model_dump():
            fw_messages.append(AssistantMessage(content=str(msg)))
        else:
            fw_messages.append(UserMessage(content=str(msg)))

    return fw_messages
