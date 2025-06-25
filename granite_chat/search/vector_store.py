"""
Wrapper for langchain vector store
Enables configurable chunk size
Add document index
"""

from typing import Any

from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import VectorStore
from transformers import AutoTokenizer

from granite_chat.search.types import ScrapedContent


class ConfigurableVectorStoreWrapper:
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

    def load(self, content: list[ScrapedContent]) -> None:
        """
        Load the documents into vector_store
        Translate to langchain doc type, split to chunks then load
        """
        langchain_documents = self._create_langchain_documents(content)
        splitted_documents = self._split_documents(langchain_documents)

        # TODO: Watsonx embedding bails out if > 1000 embedding docs, implement a better fix here
        splitted_documents = splitted_documents[0 : min(len(splitted_documents), 1000)]

        self.vector_store.add_documents(splitted_documents)

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
        results = await self.vector_store.asimilarity_search(query=query, k=k, filter=filter)
        return results
