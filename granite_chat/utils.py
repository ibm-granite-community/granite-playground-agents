from collections.abc import Generator
from typing import TypeVar

from acp_sdk import CitationMetadata, MessagePart
from acp_sdk import Message as ACPMessage
from beeai_framework.backend import AssistantMessage, UserMessage
from beeai_framework.backend import Message as FrameworkMessage
from granite_io.types import AssistantMessage as GraniteIOAssistantMessage
from granite_io.types import UserMessage as GraniteIOUserMessage

from granite_chat.citations.types import Citation


def filter_msg_parts(parts: list[MessagePart], content_type: str = "text/plain") -> list[MessagePart]:
    return [p for p in parts if p.content_type == content_type]


def to_beeai_framework_messages(messages: list[ACPMessage]) -> list[FrameworkMessage]:
    """
    Convert ACP message list to BeeAI framework message list.

    Args:
        messages (list[ACPMessage]): List of ACP messages.

    Returns:
        list[FrameworkMessage]: A list of BeeAI framework messages.
    """
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


def to_granite_io_messages(messages: list[ACPMessage]) -> list[GraniteIOAssistantMessage | GraniteIOUserMessage]:
    gio_messages: list[GraniteIOAssistantMessage | GraniteIOUserMessage] = []

    for msg in messages:
        msg_dict = msg.parts[0].model_dump()

        if "role" in msg_dict and msg_dict["role"] == "user":
            gio_messages.append(GraniteIOUserMessage(content=str(msg)))
        else:
            gio_messages.append(GraniteIOAssistantMessage(content=str(msg)))

    return gio_messages


T = TypeVar("T")


def batch(lst: list[T], batch_size: int) -> Generator[list[T], None, None]:
    """
    Yield successive batches of `batch_size` from `lst`.
    """
    length = len(lst)
    for i in range(0, length, batch_size):
        yield list(lst[i : i + batch_size])


def to_citation_message_part(citation: Citation) -> MessagePart:
    """
    Wrap a Citation object as MessagePart

    Args:
        citation (Citation): Citation object.

    Returns:
        MessagePart: A message part containing the CitationMetadata
    """
    return MessagePart(
        metadata=CitationMetadata(
            url=citation.url,
            title=citation.title,
            description=citation.context_text,
            start_index=citation.start_index,
            end_index=citation.end_index,
        ),
    )
