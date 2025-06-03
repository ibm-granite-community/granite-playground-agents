"""
Wrapper for langchain vector store
Enables configurable chunk size
"""

from gpt_researcher.vector_store import VectorStoreWrapper  # type: ignore
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import VectorStore


class GraniteVectorStoreWrapper(VectorStoreWrapper):

    def __init__(self, vector_store: VectorStore, chunk_size: int = 1000, chunk_overlap: int = 200) -> None:
        super().__init__(vector_store)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def _split_documents(
        self, documents: list[Document], chunk_size: int = 1000, chunk_overlap: int = 200
    ) -> list[Document]:
        """
        Split documents into smaller chunks
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        return text_splitter.split_documents(documents)
