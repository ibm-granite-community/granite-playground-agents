from collections.abc import AsyncIterator

from beeai_framework.backend import (
    ChatModelNewTokenEvent,
    UserMessage,
)
from beeai_framework.backend.chat import ChatModel
from langchain_core.documents import Document

from granite_chat.citations.prompts import CitationsPrompts


class CitationRegenerator:
    def __init__(self, chat_model: ChatModel) -> None:
        self.chat_model = chat_model

    async def regenerate(self, response: str, documents: list[Document]) -> AsyncIterator[str]:
        """
        Regenerate a response with inline citations.

        :param response: Original response text.
        :param documents: List of documents, each with a 'doc_id' and 'page_content'.
        :return: Yields MessagePart instances as output is streamed.
        """
        # Construct the prompt
        prompt = self._build_prompt(response, documents)

        async for data, event in self.chat_model.create(messages=[UserMessage(content=prompt)], stream=True):
            match (data, event.name):
                case (ChatModelNewTokenEvent(), "new_token"):
                    yield data.value.get_text_content()

    def _build_prompt(self, response: str, documents: list[Document]) -> str:
        return CitationsPrompts.generate_response_with_citations_prompt(response=response, docs=documents)
