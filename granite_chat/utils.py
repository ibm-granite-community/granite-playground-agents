from acp_sdk import Message as ACPMessage
from acp_sdk import MessagePart
from beeai_framework.backend import AssistantMessage, UserMessage
from beeai_framework.backend import Message as FrameworkMessage
from granite_io.types import AssistantMessage as GraniteIOAssistantMessage  # type: ignore
from granite_io.types import UserMessage as GraniteIOUserMessage


def filter_msg_parts(parts: list[MessagePart], content_type: str = "text/plain") -> list[MessagePart]:
    return [p for p in parts if p.content_type == content_type]


def to_beeai_framework(messages: list[ACPMessage]) -> list[FrameworkMessage]:
    fw_messages: list[FrameworkMessage] = []

    for msg in messages:
        parts = filter_msg_parts(msg.parts)

        if len(parts) > 0:
            msg_dict = parts[0].model_dump()
            if "role" in msg_dict and msg_dict["role"] == "user":
                fw_messages.append(UserMessage(content=str(msg)))
            else:
                fw_messages.append(AssistantMessage(content=str(msg)))

    return fw_messages


def to_granite_io(messages: list[ACPMessage]) -> list[GraniteIOAssistantMessage | GraniteIOUserMessage]:
    gio_messages: list[GraniteIOAssistantMessage | GraniteIOUserMessage] = []

    for msg in messages:
        msg_dict = msg.parts[0].model_dump()

        if "role" in msg_dict and msg_dict["role"] == "user":
            gio_messages.append(GraniteIOUserMessage(content=str(msg)))
        else:
            gio_messages.append(GraniteIOAssistantMessage(content=str(msg)))

    return gio_messages
