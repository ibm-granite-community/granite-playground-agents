import asyncio
from abc import ABC, abstractmethod
from itertools import count

import nltk
import numpy as np
from acp_sdk import Message
from beeai_framework.backend import UserMessage
from granite_io import make_backend
from granite_io.io.citations import CitationsIOProcessor
from granite_io.types import AssistantMessage as GraniteIOAssistantMessage
from granite_io.types import ChatCompletionInputs, GenerateInputs
from granite_io.types import Document as GraniteIODocument
from langchain_core.documents import Document
from nltk.tokenize import sent_tokenize
from sklearn.metrics.pairwise import cosine_similarity

from granite_chat import get_logger
from granite_chat.chat_model import ChatModelFactory
from granite_chat.citations.events import CitationEvent
from granite_chat.citations.prompts import CitationsPrompts
from granite_chat.citations.types import (
    Citation,
    CitationsSchema,
    ReferencingCitationsSchema,
    Sentence,
)
from granite_chat.config import settings
from granite_chat.emitter import EventEmitter
from granite_chat.markdown import MarkdownSection, get_markdown_sections, get_markdown_tokens_with_content
from granite_chat.search.embeddings.factory import EmbeddingsFactory
from granite_chat.work import chat_pool, task_pool

nltk.download("punkt_tab")

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
            return ReferencingMatchingCitationGenerator()


class CitationGenerator(ABC, EventEmitter):
    """Factory for generating citations"""

    @abstractmethod
    async def generate(self, messages: list[Message], docs: list[Document], response: str) -> None:
        """Generate citations"""
        pass


class GraniteIOCitationGenerator(CitationGenerator):
    """Simple sources listed in markdown."""

    def __init__(self, openai_base_url: str, model_id: str, extra_headers: dict[str, str] | None = None) -> None:
        super().__init__()
        self.openai_base_url = openai_base_url
        self.model_id = model_id
        self.extra_headers = extra_headers
        self.citations_io_processor = CitationsIOProcessor(
            backend=make_backend(
                "openai",
                {
                    "model_name": self.model_id,
                    "openai_base_url": self.openai_base_url,
                },
            )
        )

    async def generate(self, messages: list[Message], docs: list[Document], response: str) -> None:
        try:
            sections = get_markdown_sections(response)
            for section in sections:
                if section.content.strip():
                    async with task_pool.throttle():
                        await self._generate_citations(messages, docs, section)
        except asyncio.CancelledError:
            raise

    async def _generate_citations(
        self, messages: list[Message], docs: list[Document], section: MarkdownSection
    ) -> None:
        try:
            granite_io_messages = [GraniteIOAssistantMessage(content=section.content)]
            granite_io_documents = [GraniteIODocument(doc_id=str(i), text=d.page_content) for i, d in enumerate(docs)]
            doc_index = {str(i): d for i, d in enumerate(docs)}

            async with chat_pool.throttle():
                result = await self.citations_io_processor.acreate_chat_completion(
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
                    tokens = get_markdown_tokens_with_content(gio_citation.response_text)

                    for tok in tokens:
                        stripped = tok.content.strip()

                        if stripped.endswith(":") or stripped.endswith("*") or not stripped.endswith("."):
                            continue

                        start_index = gio_citation.response_begin + section.start_index + tok.start_index
                        end_index = gio_citation.response_begin + +section.start_index + tok.end_index

                        await self._emit(
                            CitationEvent(
                                citation=Citation(
                                    url=doc.metadata["url"],
                                    title=doc.metadata["title"],
                                    context_text=gio_citation.context_text,
                                    start_index=start_index,
                                    end_index=end_index,
                                )
                            )
                        )

        except Exception as e:  # Malformed citations throws error
            logger.info(f"Failed to generate citations for `{section.content}`")
            logger.exception(repr(e))


class DefaultCitationGenerator(CitationGenerator):
    """Simple sources listed in markdown."""

    def __init__(self) -> None:
        super().__init__()
        self.chat_model = ChatModelFactory.create("structured")

    async def generate(self, messages: list[Message], docs: list[Document], response: str) -> None:
        try:
            sections = get_markdown_sections(response)
            for section in sections:
                if section.content.strip():
                    async with task_pool.throttle():
                        await self._generate_citations(messages, docs, section)

        except asyncio.CancelledError:
            raise

    async def _generate_citations(
        self, messages: list[Message], docs: list[Document], section: MarkdownSection
    ) -> None:
        try:
            doc_index = {str(i): d for i, d in enumerate(docs)}
            source_index = {d.metadata["source"]: d.metadata["title"] for d in docs}
            tokens = get_markdown_tokens_with_content(section.content)
            sentences = []
            counter = count(start=0, step=1)
            for tok in tokens:
                stripped_content = tok.content.strip()
                if tok.type == "inline" and stripped_content.endswith("."):
                    sentences.extend(
                        self._to_sentences(
                            response=tok.content, offset=section.start_index + tok.start_index, counter=counter
                        )
                    )

            sent_index: dict[str, Sentence] = {str(s.id): s for s in sentences}
            prompt = CitationsPrompts.generate_citations_prompt(sentences=sentences, docs=docs)

            async with chat_pool.throttle():
                structured_output = await self.chat_model.create_structure(
                    schema=CitationsSchema, messages=[UserMessage(content=prompt)]
                )

            citations = CitationsSchema(**structured_output.object)

            for cite in citations.citations:
                if cite.sentence_id in sent_index and cite.source_id in doc_index:
                    source = doc_index[cite.source_id].metadata["source"]
                    if source in source_index:
                        sentence = sent_index[cite.sentence_id]
                        await self._emit(
                            CitationEvent(
                                citation=Citation(
                                    url=source,
                                    title=source_index[source],
                                    context_text=cite.source_summary,
                                    start_index=sentence.offset,
                                    end_index=sentence.offset + sentence.length,
                                )
                            )
                        )

        except Exception as e:
            logger.info(f"Failed to generate citations for `{section.content}`")
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


class ReferencingMatchingCitationGenerator(CitationGenerator):
    """Simple sources listed in markdown."""

    def __init__(self) -> None:
        super().__init__()
        self.chat_model = ChatModelFactory.create("structured")
        self.embeddings = EmbeddingsFactory.create()
        self.sentence_splitter = nltk.tokenize.punkt.PunktSentenceTokenizer()

    async def generate(self, messages: list[Message], docs: list[Document], response: str) -> None:
        try:
            docs_as_sentences = [list(self.sentence_splitter.tokenize(d.page_content)) for d in docs]
            docs_as_sentences_flat = [s for sub in docs_as_sentences for s in sub]
            src_embeddings = await self.embeddings.aembed_documents(docs_as_sentences_flat)
            np_src = np.atleast_2d(np.array(src_embeddings))

            doc_sentence_offsets = [list(self.sentence_splitter.span_tokenize(d.page_content)) for d in docs]
            flat_doc_sentence_offsets = [offset for offsets in doc_sentence_offsets for offset in offsets]
            sentence_to_doc: list[int] = []

            for doc_ix, doc_offsets in enumerate(doc_sentence_offsets):
                sentence_to_doc = sentence_to_doc + ([doc_ix] * len(doc_offsets))

            sections = get_markdown_sections(response)

            for section in sections:
                if section.content.strip():
                    async with task_pool.throttle():
                        counter = count(start=0, step=1)
                        tokens = get_markdown_tokens_with_content(section.content)
                        response_as_sentences: list[Sentence] = []
                        for tok in tokens:
                            stripped_content = tok.content.strip()
                            if tok.type == "inline" and stripped_content.endswith("."):
                                response_as_sentences.extend(
                                    self._to_sentences(
                                        response=tok.content,
                                        offset=section.start_index + tok.start_index,
                                        counter=counter,
                                    )
                                )

                        if len(response_as_sentences):
                            response_embeddings = await self.embeddings.aembed_documents(
                                [s.text for s in response_as_sentences]
                            )
                            np_resp = np.atleast_2d(np.array(response_embeddings))
                            sim_matrix = cosine_similarity(np_resp, np_src)
                            top_n_indices = np.argsort(sim_matrix, axis=1)[:, -settings.CITATIONS_MAX_STATEMENTS :][
                                :, ::-1
                            ]  # sort descending
                            top_n_scores = np.take_along_axis(sim_matrix, top_n_indices, axis=1)

                            flat_indices = top_n_indices.flatten()
                            flat_scores = top_n_scores.flatten()

                            # Sort flattened indices by score descending
                            sorted_flat_indices = flat_indices[np.argsort(-flat_scores)]

                            src_indices: set[int] = set(sorted_flat_indices.tolist())
                            rewritten_docs = [f"<s{i}> {docs_as_sentences_flat[i]}" for i in src_indices]
                            response_list = [f"<r{s.id}> {s.text}" for s in response_as_sentences]

                            prompt = CitationsPrompts.generate_references_citations_prompt(
                                response=response_list, docs=rewritten_docs
                            )

                            async with chat_pool.throttle():
                                structured_output = await self.chat_model.create_structure(
                                    schema=ReferencingCitationsSchema, messages=[UserMessage(content=prompt)]
                                )

                            citations = ReferencingCitationsSchema(**structured_output.object)
                            message_sentence_offsets = [(s.offset, s.offset + s.length) for s in response_as_sentences]

                            for cite in citations.citations:
                                if 0 <= cite.r < len(message_sentence_offsets):
                                    response_begin, response_end = message_sentence_offsets[cite.r]
                                    if cite.s in src_indices:
                                        doc_num = sentence_to_doc[cite.s]
                                        context_begin, context_end = flat_doc_sentence_offsets[cite.s]
                                        context_text = docs[doc_num].page_content[context_begin:context_end]

                                        await self._emit(
                                            CitationEvent(
                                                citation=Citation(
                                                    url=docs[doc_num].metadata["url"],
                                                    title=docs[doc_num].metadata["title"],
                                                    context_text=context_text,
                                                    start_index=response_begin,
                                                    end_index=response_end,
                                                )
                                            )
                                        )

        except asyncio.CancelledError:
            raise

    async def _generate_citations(
        self, messages: list[Message], docs: list[Document], section: MarkdownSection
    ) -> None:
        try:
            doc_index = {str(i): d for i, d in enumerate(docs)}
            source_index = {d.metadata["source"]: d.metadata["title"] for d in docs}
            tokens = get_markdown_tokens_with_content(section.content)
            sentences = []
            counter = count(start=0, step=1)
            for tok in tokens:
                stripped_content = tok.content.strip()
                if tok.type == "inline" and stripped_content.endswith("."):
                    sentences.extend(
                        self._to_sentences(
                            response=tok.content, offset=section.start_index + tok.start_index, counter=counter
                        )
                    )

            sent_index: dict[str, Sentence] = {str(s.id): s for s in sentences}
            prompt = CitationsPrompts.generate_citations_prompt(sentences=sentences, docs=docs)

            async with chat_pool.throttle():
                structured_output = await self.chat_model.create_structure(
                    schema=CitationsSchema, messages=[UserMessage(content=prompt)]
                )

            citations = CitationsSchema(**structured_output.object)

            for cite in citations.citations:
                if cite.sentence_id in sent_index and cite.source_id in doc_index:
                    source = doc_index[cite.source_id].metadata["source"]
                    if source in source_index:
                        sentence = sent_index[cite.sentence_id]
                        await self._emit(
                            CitationEvent(
                                citation=Citation(
                                    url=source,
                                    title=source_index[source],
                                    context_text="",
                                    start_index=sentence.offset,
                                    end_index=sentence.offset + sentence.length,
                                )
                            )
                        )

        except Exception as e:
            logger.info(f"Failed to generate citations for `{section.content}`")
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
