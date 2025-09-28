from langchain_core.vectorstores import InMemoryVectorStore

from granite_core.config import settings
from granite_core.search.embeddings.factory import EmbeddingsFactory
from granite_core.search.embeddings.model import EmbeddingsModel
from granite_core.search.vector_store import VectorStoreWrapper


class VectorStoreWrapperFactory:
    """Factory for VectorStore instances."""

    @staticmethod
    def create() -> VectorStoreWrapper:
        embeddings_model: EmbeddingsModel = EmbeddingsFactory.create(model_type="retrieval")
        lc_vector_store = InMemoryVectorStore(embedding=embeddings_model.embeddings)

        if embeddings_model.tokenizer:
            vector_store_wrapper = VectorStoreWrapper(
                vector_store=lc_vector_store,
                chunk_size=settings.CHUNK_SIZE - 2,
                chunk_overlap=settings.CHUNK_OVERLAP,
                tokenizer=embeddings_model.tokenizer,
            )
        else:
            # Fall back on character chunks
            vector_store_wrapper = VectorStoreWrapper(
                vector_store=lc_vector_store,
                chunk_size=settings.CHUNK_SIZE,
                chunk_overlap=settings.CHUNK_OVERLAP,
            )

        return vector_store_wrapper
