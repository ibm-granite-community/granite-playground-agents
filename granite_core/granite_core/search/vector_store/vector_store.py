# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0


# Portions of this file are derived from the Apache 2.0 licensed project "gpt-researcher"
# Original source: https://github.com/assafelovic/gpt-researcher
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Changes made:
# - Configurable chunk size
# - Add document index


import asyncio
from functools import partial
from typing import Any

from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import EmbeddingsFilter
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_classic.vectorstores import VectorStore
from langchain_core.documents import Document
from transformers import AutoTokenizer

from granite_core.config import settings
from granite_core.search.scraping.types import ScrapedSearchResult
from granite_core.work import task_pool


class VectorStoreWrapper:
    def __init__(
        self,
        vector_store: VectorStore,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        tokenizer: AutoTokenizer | None = None,
    ) -> None:
        self.vector_store = vector_store
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.tokenizer = tokenizer

    async def load(self, content: list[ScrapedSearchResult]) -> None:
        """
        Load the documents into vector_store
        Translate to langchain doc type, split to chunks then load
        """
        langchain_documents = self._create_langchain_documents(content)
        splitted_documents = await self._a_split_documents(langchain_documents)
        await self.vector_store.aadd_documents(splitted_documents)

    # TODO: subclass Document for better typing support
    def _create_langchain_documents(self, scraped_content: list[ScrapedSearchResult]) -> list[Document]:
        return [
            Document(
                page_content=item.raw_content if item.raw_content else "",
                metadata={
                    "source": item.search_result.url,
                    "index": i,
                    "url": item.search_result.url,
                    "title": item.search_result.title,
                    "snippet": item.search_result.snippet,
                },
            )
            for i, item in enumerate(scraped_content)
        ]

    async def _a_split_documents(self, documents: list[Document]) -> list[Document]:
        split_docs: list[Document] = []
        loop = asyncio.get_running_loop()

        for doc in documents:
            splitted = await asyncio.wait_for(
                loop.run_in_executor(
                    task_pool.executor,
                    partial(self._split_documents, [doc]),
                ),
                timeout=None,
            )
            split_docs.extend(splitted)
            await asyncio.sleep(0)

        return split_docs

    def _split_documents(self, documents: list[Document]) -> list[Document]:
        """
        Split documents into smaller chunks
        """
        if self.tokenizer:
            text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
                tokenizer=self.tokenizer,  # type: ignore
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            )
        else:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            )

        return text_splitter.split_documents(documents)

    async def asimilarity_search(self, query: str, k: int, filter: dict[str, Any] | None = None) -> list[Document]:
        """Return query by vector store"""

        if self.vector_store and self.vector_store.embeddings:
            retriever = self.vector_store.as_retriever(
                search_type="mmr", search_kwargs={"k": k, "lambda_mult": settings.MMR_LAMBDA_MULT}
            )

            embeddings_filter = EmbeddingsFilter(embeddings=self.vector_store.embeddings, similarity_threshold=0.65)
            compression_retriever = ContextualCompressionRetriever(
                base_compressor=embeddings_filter, base_retriever=retriever
            )
            results = await compression_retriever.ainvoke(input=query)
            return results
        else:
            raise ValueError("Embeddings must not be None")
