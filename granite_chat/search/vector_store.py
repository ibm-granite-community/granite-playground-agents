"""
Wrapper for langchain vector store
Enables configurable chunk size
Add document index
"""

from gpt_researcher.vector_store import VectorStoreWrapper  # type: ignore
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import VectorStore
from transformers import AutoTokenizer


class ConfigurableVectorStoreWrapper(VectorStoreWrapper):
    def __init__(
        self,
        vector_store: VectorStore,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        tokenizer: AutoTokenizer | None = None,
    ) -> None:
        super().__init__(vector_store)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.tokenizer = tokenizer

    def _create_langchain_documents(self, data: list[dict]) -> list[Document]:
        return [
            Document(page_content=item["raw_content"], metadata={"source": item["url"], "index": i})
            for i, item in enumerate(data)
        ]

    def _split_documents(
        self, documents: list[Document], chunk_size: int = 1000, chunk_overlap: int = 200
    ) -> list[Document]:
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
