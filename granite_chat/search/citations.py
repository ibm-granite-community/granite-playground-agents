import traceback
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator

from acp_sdk import Message, MessagePart
from granite_io import make_backend  # type: ignore
from granite_io.io.citations import CitationsIOProcessor  # type: ignore
from granite_io.types import AssistantMessage as GraniteIOAssistantMessage  # type: ignore
from granite_io.types import ChatCompletionInputs, GenerateInputs
from granite_io.types import Document as GraniteIODocument
from langchain_core.documents import Document

from granite_chat.search.types import Citation, Source
from granite_chat.utils import to_granite_io


class CitationGenerator(ABC):
    """Factory for generating citations"""

    @abstractmethod
    def generate(
        self, messages: list[Message], docs: list[Document], response: str
    ) -> AsyncGenerator[MessagePart, None]:
        """Generate citations"""
        pass


class DefaultCitationGenerator(CitationGenerator):
    """Simple sources listed in markdown."""

    async def generate(
        self, messages: list[Message], docs: list[Document], response: str
    ) -> AsyncGenerator[MessagePart, None]:
        sources = {Source(url=doc.metadata["url"], title=doc.metadata["title"]) for doc in docs}

        yield MessagePart(content_type="source", content="\n\n**Sources:**\n", role="assistant")  # type: ignore[call-arg]

        for i, source in enumerate(sources):
            doc_str = f"{i + 1!s}. [{source.title}]({source.url})\n"
            yield MessagePart(content_type="source", content=doc_str, role="assistant")  # type: ignore[call-arg]


class GraniteIOCitationGenerator(CitationGenerator):
    """Simple sources listed in markdown."""

    def __init__(self, openai_base_url: str, model_id: str, extra_headers: dict[str, str] | None = None) -> None:
        super().__init__()
        self.openai_base_url = openai_base_url
        self.model_id = model_id
        self.extra_headers = extra_headers

    async def generate(
        self, messages: list[Message], docs: list[Document], response: str
    ) -> AsyncGenerator[MessagePart, None]:
        citations_io_processor = CitationsIOProcessor(
            backend=make_backend(
                "openai",
                {
                    "model_name": self.model_id,
                    "openai_base_url": self.openai_base_url,
                },
            )
        )

        try:
            granite_io_messages = to_granite_io(messages)
            # Add agent response
            granite_io_messages.append(GraniteIOAssistantMessage(content=response))
            granite_io_documents = [GraniteIODocument(doc_id=str(i), text=d.page_content) for i, d in enumerate(docs)]
            doc_index = {str(i): d for i, d in enumerate(docs)}

            result = await citations_io_processor.acreate_chat_completion(
                ChatCompletionInputs(
                    messages=granite_io_messages,
                    documents=granite_io_documents,
                    controls={"citations": True},
                    generate_inputs=GenerateInputs(temperature=0.0, extra_headers=self.extra_headers),
                )
            )

            result = result.results[0].next_message

            for gio_citation in result.citations:
                # Append additional metadata

                original_doc = doc_index[gio_citation.doc_id]

                citation = Citation(
                    citation_id=gio_citation.citation_id,
                    document_id=gio_citation.doc_id,
                    source=original_doc.metadata["url"],
                    source_title=original_doc.metadata["title"],
                    context_text=gio_citation.context_text,
                    context_begin=gio_citation.context_begin,
                    context_end=gio_citation.context_end,
                    response_text=gio_citation.response_text,
                    response_begin=gio_citation.response_begin,
                    response_end=gio_citation.response_end,
                )

                yield MessagePart(content_type="source/citation", content=citation.model_dump_json(), role="assistant")  # type: ignore[call-arg]

        except Exception:  # Malformed citations throws error
            traceback.print_exc()
