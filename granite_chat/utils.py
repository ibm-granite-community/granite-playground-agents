from acp_sdk import Message as ACPMessage
from beeai_framework.backend import AssistantMessage, UserMessage
from beeai_framework.backend import Message as FrameworkMessage


def to_beeai_framework(messages: list[ACPMessage]) -> list[FrameworkMessage]:
    fw_messages: list[FrameworkMessage] = []

    for msg in messages:
        msg_dict = msg.parts[0].model_dump()

        if "role" in msg_dict and msg_dict["role"] == "user":
            fw_messages.append(UserMessage(content=str(msg)))
        else:
            fw_messages.append(AssistantMessage(content=str(msg)))

    return fw_messages
