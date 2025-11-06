# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0


import asyncio

import pytest
from httpx import AsyncClient

from granite_core.search.scraping.docling import DoclingPDFScraper
from granite_core.search.scraping.types import ScrapedContent


@pytest.fixture
def client() -> AsyncClient:
    return AsyncClient()


@pytest.fixture
def scraper() -> DoclingPDFScraper:
    return DoclingPDFScraper()


@pytest.mark.asyncio
async def test_docling_pdf(scraper: DoclingPDFScraper, client: AsyncClient) -> None:
    """Test basic pdf scraping"""
    content: ScrapedContent | None = await asyncio.wait_for(
        scraper.ascrape(
            link="https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf",
            client=client,
        ),
        timeout=10,
    )

    assert content is not None
    assert content.title == "Attention Is All You Need"
    assert len(content.content) > 0


@pytest.mark.asyncio
async def test_docling_pdf_timeout(scraper: DoclingPDFScraper, client: AsyncClient) -> None:
    """Test pdf scraping timeout"""
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(
            scraper.ascrape(link="https://hai.stanford.edu/assets/files/hai_ai_index_report_2025.pdf", client=client),
            timeout=10,
        )
