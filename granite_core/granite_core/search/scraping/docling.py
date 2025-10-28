# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0


import asyncio
import os
import tempfile
from urllib.parse import urlparse

import aiofiles
from docling_core.types.doc.page import TextCellUnit
from docling_parse.pdf_parser import DoclingPdfParser, PdfDocument
from httpx import AsyncClient, TimeoutException

from granite_core.logging import get_logger
from granite_core.search.scraping.scraper import AsyncScraper

logger = get_logger(__name__)


class DoclingPDFScraper(AsyncScraper):
    def is_url(self, link: str) -> bool:
        """
        Check if the provided `link` is a valid URL.

        Returns:
          bool: True if the link is a valid URL, False otherwise.
        """
        try:
            result = urlparse(link)
            return all([result.scheme, result.netloc])  # Check for valid scheme and network location
        except Exception:
            return False

    async def ascrape(self, link: str, client: AsyncClient) -> tuple:
        parser = DoclingPdfParser()
        temp_filename: str | None = None

        try:
            if self.is_url(link):
                # Download to local
                async with client.stream("GET", link, timeout=8) as response:
                    response.raise_for_status()

                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                        temp_filename = temp_file.name

                    async with aiofiles.open(temp_filename, "wb") as f:
                        async for chunk in response.aiter_bytes(chunk_size=8192):
                            await f.write(chunk)
                path = temp_filename
            else:
                # Assumes a local path
                path = link

            pdf_doc: PdfDocument = await asyncio.to_thread(parser.load, path)
            lines: list[str] = []

            for _, pred_page in pdf_doc.iterate_pages():
                for word in pred_page.iterate_cells(unit_type=TextCellUnit.LINE):
                    lines.append(word.text)

            content = "\n".join(lines)
            return content, [], lines[0] if lines else ""

        except TimeoutException:
            logger.exception(f"Download timed out. Please check the link: {link}")
            return "", [], ""
        except Exception:
            logger.exception(f"Error loading PDF: {link}")
            return "", [], ""
        finally:
            if temp_filename and os.path.exists(temp_filename):
                os.remove(temp_filename)
