# agent.py
import os

from collections.abc import AsyncGenerator

from acp_sdk import MessagePart, Metadata
from acp_sdk.models import Message
from acp_sdk.server import Context, Server

from beeai_framework.adapters.openai import OpenAIChatModel
from beeai_framework.backend import ChatModelNewTokenEvent, ChatModelParameters

from granite_chat import utils
from dotenv import load_dotenv

load_dotenv()

MODEL_NAME = os.environ["MODEL_NAME"]
OPENAI_URL = os.environ["OPENAI_URL"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
OPENAI_API_HEADERS = os.environ["OPENAI_API_HEADERS"]

MAX_TOKENS = 4096
TEMPERATURE = 0.3

server = Server()


@server.agent(
    name="granite-chat",
    description="Granite Chat",
    metadata=Metadata(ui={"type": "chat"}),  # type: ignore[call-arg]
)
async def granite_chat(input: list[Message], context: Context) -> AsyncGenerator:

    model = OpenAIChatModel(
        model_id=MODEL_NAME,
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_URL,
        parameters=ChatModelParameters(max_tokens=MAX_TOKENS, temperature=TEMPERATURE),
    )

    # TODO: Manage context window
    messages = utils.to_beeai_framework(messages=input)

    async for data, event in model.create(messages=messages, stream=True):
        match (data, event.name):
            case (ChatModelNewTokenEvent(), "new_token"):
                yield MessagePart(content_type="text/plain", content=data.value.get_text_content(), role="assistant")  # type: ignore[call-arg]


server.run()
