# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0


import pytest

from granite_core.search.embeddings.factory import EmbeddingsFactory
from granite_core.search.embeddings.model import EmbeddingsModel


@pytest.mark.asyncio
async def test_basic_embeddings() -> None:
    """Test basic embedding infrastructure"""
    embeddings_model: EmbeddingsModel = EmbeddingsFactory.create()
    vectors: list[list[float]] = await embeddings_model.embeddings.aembed_documents(texts=["King", "Queen"])
    assert vectors and len(vectors) == 2


@pytest.mark.asyncio
async def test_bad_encoding() -> None:
    """Test embedding strings that are known to cause problems"""
    embeddings_model: EmbeddingsModel = EmbeddingsFactory.create()
    bad_texts: list[str] = [
        'table.compare th{background:#f3f4f6;font-weight:700}\\n#qmeta-collection table.compare tr:last-child td{border-bottom:none}\\n#qmeta-collection .cols{columns:2;gap:28px}\\n@media(max-width:700px){#qmeta-collection .cols{columns:1}}\\n#qmeta-collection details{background:#f9fafb;border:1px solid var(--line);border-radius:14px;padding:10px 12px}\\n#qmeta-collection details+details{margin-top:10px}\\n#qmeta-collection details summary{cursor:pointer;font-weight:700;color:var(--ink)}\\n#qmeta-collection .faq dl{display:grid;grid-template-columns:1fr;gap:10px}\\n#qmeta-collection .faq dt{font-weight:800;color:var(--ink)}\\n#qmeta-collection .faq dt strong{color:var(--ink)}\\n#qmeta-collection .note{font-size:.9rem;color:var(--muted)}\\n#qmeta-collection footer{padding:30px 0;color:var(--muted)}\\n#qmeta-collection code.k{background:#f3f4f6;border:1px solid var(--line);padding:2px 6px;border-radius:6px}\\n\\u003c\\/style\\u003e\\n\\u003cdiv class=\\"wrap\\" id=\\"qmeta-collection\\"\\u003e\\n\\u003c!-- HERO --\\u003e\\u003cheader class=\\"hero\\"\\u003e\\n\\u003cdiv class=\\"kicker\\"\\u003eKEF Q Series Speakers\\u003c\\/div\\u003e\\n\\u003ch2\\u003eKEF Q11, Q7, Q Concerto, Q3, Q1, Q6, Q8 \\u0026amp; Q4 \u2014 Compare \\u0026amp; Shop\\u003c\\/h2\\u003e\\n\\u003cp'  # noqa: E501
    ]
    vectors: list[list[float]] = await embeddings_model.embeddings.aembed_documents(texts=bad_texts)
    assert len(vectors) == len(bad_texts)
