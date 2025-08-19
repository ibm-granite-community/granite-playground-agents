"""
Wrapper for langchain vector store
Enables configurable chunk size
Add document index
"""

import asyncio
from typing import Any

from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import VectorStore
from langchain_core.documents import Document
from transformers import AutoTokenizer

from granite_chat.config import settings
from granite_chat.search.types import ScrapedContent
from granite_chat.work import task_pool


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

    async def load(self, content: list[ScrapedContent]) -> None:
        """
        Load the documents into vector_store
        Translate to langchain doc type, split to chunks then load
        """
        langchain_documents = self._create_langchain_documents(content)

        loop = asyncio.get_running_loop()

        splitted_documents = await asyncio.wait_for(
            loop.run_in_executor(task_pool.executor, lambda: self._split_documents(langchain_documents)), timeout=None
        )

        await self.vector_store.aadd_documents(splitted_documents)

    # TODO: subclass Document for better typing support
    def _create_langchain_documents(self, scraped_content: list[ScrapedContent]) -> list[Document]:
        return [
            Document(
                page_content=item.raw_content if item.raw_content else "",
                metadata={
                    "source": item.search_result.url,
                    "index": i,
                    "url": item.search_result.url,
                    "title": item.search_result.title,
                    "snippet": item.search_result.body,
                },
            )
            for i, item in enumerate(scraped_content)
        ]

    def _split_documents(self, documents: list[Document]) -> list[Document]:
        """
        Split documents into smaller chunks
        """
        if self.tokenizer:
            text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
                tokenizer=self.tokenizer,
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

            embeddings_filter = EmbeddingsFilter(embeddings=self.vector_store.embeddings, similarity_threshold=0.5)
            compression_retriever = ContextualCompressionRetriever(
                base_compressor=embeddings_filter, base_retriever=retriever
            )
            results = await compression_retriever.ainvoke(input=query)
            return results
        else:
            raise ValueError("Embeddings must not be None")
