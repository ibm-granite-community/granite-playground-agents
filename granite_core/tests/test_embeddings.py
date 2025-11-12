# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0


import pytest

from granite_core.search.embeddings.factory import EmbeddingsFactory
from granite_core.search.embeddings.model import EmbeddingsModel


@pytest.mark.asyncio
async def test_basic_embeddings() -> None:
    """Test basic embedding infrastructure"""
    embeddings_model: EmbeddingsModel = EmbeddingsFactory.create()
    vectors = await embeddings_model.embeddings.aembed_documents(["King", "Queen"])
    assert vectors and len(vectors) == 2
