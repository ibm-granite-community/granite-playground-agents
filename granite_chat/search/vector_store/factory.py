from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore

from granite_chat.config import settings
from granite_chat.search.embeddings.factory import EmbeddingsFactory
from granite_chat.search.embeddings.tokenizer import EmbeddingsTokenizer
from granite_chat.search.vector_store import VectorStoreWrapper


class VectorStoreWrapperFactory:
    """Factory for VectorStore instances."""

    @staticmethod
    def create() -> VectorStoreWrapper:
        embeddings: Embeddings = EmbeddingsFactory.create()

        lc_vector_store = InMemoryVectorStore(embedding=embeddings)

        if settings.EMBEDDINGS_HF_TOKENIZER and (tokenizer := EmbeddingsTokenizer.get_instance().get_tokenizer()):
            vector_store_wrapper = VectorStoreWrapper(
                vector_store=lc_vector_store,
                chunk_size=settings.CHUNK_SIZE - 2,  # minus start/end tokens
                chunk_overlap=int(settings.CHUNK_OVERLAP),
                tokenizer=tokenizer,
            )
        else:
            # Fall back on character chunks
            vector_store_wrapper = VectorStoreWrapper(
                vector_store=lc_vector_store,
                chunk_size=settings.CHUNK_SIZE,
                chunk_overlap=settings.CHUNK_OVERLAP,
            )

        return vector_store_wrapper
