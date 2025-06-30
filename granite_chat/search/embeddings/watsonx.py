import asyncio

from beeai_framework.backend import EmbeddingModel, EmbeddingModelOutput
from langchain_core.embeddings import Embeddings


class WatsonxEmbeddings(Embeddings):
    """Watsonx embedding model integration."""

    def __init__(self, model_id: str) -> None:
        # TODO: use watsonx embedding model directly
        self.embedding_model = EmbeddingModel.from_name("watsonx:" + model_id)

    async def _embed(self, texts: list[str]) -> list[list[float]]:
        """Perform embedding."""
        response: EmbeddingModelOutput = await self.embedding_model.create(texts)
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
