from gpt_researcher.context.compression import VectorstoreCompressor  # type: ignore
from langchain_core.documents import Document


class CustomVectorstoreCompressor(VectorstoreCompressor):
    """Custom compressor to provide access to the context docs directly"""

    async def async_get_context_docs(self, query: str, max_results: int = 5) -> list[Document]:
        """Get relevant context from vector store"""
        return await self.vector_store.asimilarity_search(query=query, k=max_results, filter=self.filter)
