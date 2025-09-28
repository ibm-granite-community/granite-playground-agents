import asyncio
from itertools import chain
from typing import Any

from beeai_framework.backend import EmbeddingModel, EmbeddingModelOutput
from langchain_core.embeddings import Embeddings

from granite_core.config import settings
from granite_core.utils import batch
from granite_core.work import WorkerPool


class WatsonxEmbeddings(Embeddings):
    """Watsonx embedding model integration."""

    def __init__(self, model_id: str, worker_pool: WorkerPool, **kwargs: Any) -> None:
        # TODO: use watsonx embedding model directly
        self.embedding_model = EmbeddingModel.from_name("watsonx:" + model_id, settings=kwargs)
        self.worker_pool = worker_pool

    async def _embed(self, texts: list[str]) -> list[list[float]]:
        """Perform embedding."""
        # TODO: Externalize batch size
        tasks = [self._embed_doc_batch(docs) for docs in batch(texts, settings.MAX_EMBEDDINGS_PER_REQUEST)]
        results = await asyncio.gather(*tasks)
        return list(chain.from_iterable(results))

    async def _embed_doc_batch(self, texts: list[str]) -> list[list[float]]:
        async with self.worker_pool.throttle():
            response: EmbeddingModelOutput = await self.embedding_model.create(texts, max_retries=settings.MAX_RETRIES)
            return response.embeddings

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed search docs."""
        return asyncio.run(self._embed(texts))

    def embed_query(self, text: str) -> list[float]:
        """Embed query text."""
        return asyncio.run(self._embed([text]))[0]

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed search docs."""
        return await self._embed(texts)

    async def aembed_query(self, text: str) -> list[float]:
        """Embed query text."""
        embeddings = await self._embed([text])
        return embeddings[0]
