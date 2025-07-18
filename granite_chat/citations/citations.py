from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator

from acp_sdk import Message, MessagePart
from acp_sdk.models.models import CitationMetadata
from beeai_framework.backend import UserMessage
from granite_io import make_backend
from granite_io.io.citations import CitationsIOProcessor
from granite_io.types import AssistantMessage as GraniteIOAssistantMessage
from granite_io.types import ChatCompletionInputs, GenerateInputs
from granite_io.types import Document as GraniteIODocument
from langchain_core.documents import Document
from nltk.tokenize import sent_tokenize

from granite_chat import get_logger
from granite_chat.citations.prompts import CitationsPrompts
from granite_chat.citations.types import CitationsSchema, Sentence
from granite_chat.config import settings
from granite_chat.model import ChatModelFactory
from granite_chat.search.types import Citation, Source
from granite_chat.utils import to_granite_io

logger = get_logger(__name__)


class CitationGenerator(ABC):
    """Factory for generating citations"""

    @staticmethod
    def create() -> "CitationGenerator":
        if settings.GRANITE_IO_OPENAI_API_BASE and settings.GRANITE_IO_CITATIONS_MODEL_ID:
            extra_headers = (
                dict(pair.split("=", 1) for pair in settings.GRANITE_IO_OPENAI_API_HEADERS.strip('"').split(","))
                if settings.GRANITE_IO_OPENAI_API_HEADERS
                else None
            )

            return GraniteIOCitationGenerator(
                openai_base_url=str(settings.GRANITE_IO_OPENAI_API_BASE),
                model_id=settings.GRANITE_IO_CITATIONS_MODEL_ID,
                extra_headers=extra_headers,
            )
        else:
            return GraniteCitationGenerator()
            # return DefaultCitationGenerator()

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
        sources = {
            Source(url=doc.metadata["url"], title=doc.metadata["title"], snippet=doc.metadata["snippet"])
            for doc in docs
        }

        for source in sources:
            yield MessagePart(
                metadata=CitationMetadata(
                    url=source.url,
                    title=source.title,
                    start_index=len(response),
                    end_index=len(response),
                ),
            )


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

            # Map url to doc id
            url_to_id = {}
            next_id = 0

            for doc in docs:
                url = doc.metadata["source"]
                if url not in url_to_id:
                    url_to_id[url] = next_id
                    next_id += 1
                doc.metadata["doc_id"] = url_to_id[url]

            granite_io_documents = [
                GraniteIODocument(doc_id=str(d.metadata["doc_id"]), text=d.page_content) for d in docs
            ]

            for d in granite_io_documents:
                print(d.model_dump_json())

            doc_index = {str(d.metadata["doc_id"]): d for d in docs}

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

                yield MessagePart(
                    metadata=CitationMetadata(
                        url=citation.source,
                        title=citation.source_title,
                        start_index=citation.response_begin,
                        end_index=citation.response_end,
                    )
                )

        except Exception as e:  # Malformed citations throws error
            logger.exception(repr(e))


class GraniteCitationGenerator(CitationGenerator):
    """Simple sources listed in markdown."""

    def __init__(self) -> None:
        super().__init__()

    async def generate(
        self, messages: list[Message], docs: list[Document], response: str
    ) -> AsyncGenerator[MessagePart, None]:
        try:
            model = ChatModelFactory.create(provider=settings.LLM_PROVIDER)
            doc_index = {str(i): d for i, d in enumerate(docs)}
            source_index = {d.metadata["source"]: d.metadata["title"] for d in docs}

            sentences = sent_tokenize(response)
            offsets = []
            start = 0
            for i, sentence in enumerate(sentences):
                start_index = response.find(sentence, start)
                length = len(sentence)
                offsets.append(Sentence(id=str(i), text=sentence, offset=start_index, length=length))
                start = start_index + length  # advance to avoid matching earlier sentence again
            sent_index: dict[str, Sentence] = {str(s.id): s for s in offsets}

            prompt = CitationsPrompts.generate_citations_prompt(sentences=offsets, docs=docs)

            structured_output = await model.create_structure(
                schema=CitationsSchema, messages=[UserMessage(content=prompt)]
            )

            citations = CitationsSchema(**structured_output.object)

            for cite in citations.citations:
                sentence_id = cite.sentence_id
                doc_ids = cite.doc_ids
                sources = {doc_index[id].metadata["source"] for id in doc_ids}

                for source in sources:
                    if sentence_id in sent_index:
                        yield MessagePart(
                            metadata=CitationMetadata(
                                url=source,
                                title=source_index[source],
                                start_index=sent_index[sentence_id].offset,
                                end_index=sent_index[sentence_id].offset + sent_index[sentence_id].length,
                            )
                        )

        except Exception as e:  # Malformed citations throws error
            logger.exception(repr(e))
