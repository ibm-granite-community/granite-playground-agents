from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from itertools import count

from acp_sdk import Message
from beeai_framework.backend import UserMessage
from granite_io import make_backend
from granite_io.io.citations import CitationsIOProcessor
from granite_io.types import AssistantMessage as GraniteIOAssistantMessage
from granite_io.types import ChatCompletionInputs, GenerateInputs
from granite_io.types import Document as GraniteIODocument
from langchain_core.documents import Document
from nltk.tokenize import sent_tokenize

from granite_chat import get_logger
from granite_chat.chat import ChatModelService
from granite_chat.citations.prompts import CitationsPrompts
from granite_chat.citations.types import Citation, CitationsSchema, Sentence
from granite_chat.config import settings
from granite_chat.markdown import get_markdown_tokens
from granite_chat.utils import to_granite_io_messages

logger = get_logger(__name__)


class CitationGeneratorFactory:
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
            return DefaultCitationGenerator()


class CitationGenerator(ABC):
    """Factory for generating citations"""

    @abstractmethod
    def generate(self, messages: list[Message], docs: list[Document], response: str) -> AsyncGenerator[Citation, None]:
        """Generate citations"""
        pass


class GraniteIOCitationGenerator(CitationGenerator):
    """Simple sources listed in markdown."""

    def __init__(self, openai_base_url: str, model_id: str, extra_headers: dict[str, str] | None = None) -> None:
        super().__init__()
        self.openai_base_url = openai_base_url
        self.model_id = model_id
        self.extra_headers = extra_headers

    async def generate(
        self, messages: list[Message], docs: list[Document], response: str
    ) -> AsyncGenerator[Citation, None]:
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
            # TODO: Should this be just the last user message?
            granite_io_messages = to_granite_io_messages(messages)
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
                if gio_citation.doc_id in doc_index:
                    doc = doc_index[gio_citation.doc_id]

                    # Any citation that contains markdown formatting or spans lines needs to be adjusted
                    tokens = get_markdown_tokens(gio_citation.response_text)

                    for tok in tokens:
                        stripped = tok.content.strip()

                        if stripped.endswith(":") or stripped.endswith("*") or not stripped.endswith("."):
                            continue

                        start_index = gio_citation.response_begin + tok.start_index
                        end_index = gio_citation.response_begin + tok.end_index

                        yield Citation(
                            url=doc.metadata["url"],
                            title=doc.metadata["title"],
                            context_text=gio_citation.context_text,
                            start_index=start_index,
                            end_index=end_index,
                        )

        except Exception as e:  # Malformed citations throws error
            logger.exception(repr(e))

            # Falls back on the default generator in the event of failure
            fall_back_generator = DefaultCitationGenerator()

            async for citation in fall_back_generator.generate(messages=messages, docs=docs, response=response):
                yield citation


class DefaultCitationGenerator(CitationGenerator):
    """Simple sources listed in markdown."""

    def __init__(self) -> None:
        super().__init__()

    async def generate(
        self, messages: list[Message], docs: list[Document], response: str
    ) -> AsyncGenerator[Citation, None]:
        try:
            model = ChatModelService()
            doc_index = {str(i): d for i, d in enumerate(docs)}
            source_index = {d.metadata["source"]: d.metadata["title"] for d in docs}

            tokens = get_markdown_tokens(response)
            sentences = []
            counter = count(start=0, step=1)
            for tok in tokens:
                stripped_content = tok.content.strip()
                if tok.type == "inline" and stripped_content.endswith("."):
                    sentences.extend(self._to_sentences(response=tok.content, offset=tok.start_index, counter=counter))

            sent_index: dict[str, Sentence] = {str(s.id): s for s in sentences}

            prompt = CitationsPrompts.generate_citations_prompt(sentences=sentences, docs=docs)

            structured_output = await model.create_structure(
                schema=CitationsSchema, messages=[UserMessage(content=prompt)]
            )

            citations = CitationsSchema(**structured_output.object)

            for cite in citations.citations:
                if cite.sentence_id in sent_index and cite.source_id in doc_index:
                    source = doc_index[cite.source_id].metadata["source"]
                    if source in source_index:
                        sentence = sent_index[cite.sentence_id]
                        yield Citation(
                            url=source,
                            title=source_index[source],
                            context_text=cite.source_summary,
                            start_index=sentence.offset,
                            end_index=sentence.offset + sentence.length,
                        )

        except Exception as e:
            logger.exception(repr(e))

    def _to_sentences(self, response: str, offset: int, counter: count) -> list[Sentence]:
        sentences = sent_tokenize(response)
        results = []
        start = 0
        for sentence in sentences:
            start_index = response.find(sentence, start)
            length = len(sentence)

            results.append(Sentence(id=str(next(counter)), text=sentence, offset=offset + start_index, length=length))
            start = start_index + length  # advance to avoid matching earlier sentence again

        return results
