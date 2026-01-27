# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0


from pydantic import BaseModel

from granite_core.search.types import SearchResult


class ScrapedContent(BaseModel):
    url: str
    content: str
    title: str
    images: list[str] = []


class ImageUrl(BaseModel):
    score: float
    url: str


class ScrapedSearchResult(BaseModel):
    search_result: SearchResult
    url: str
    title: str | None = None
    raw_content: str | None = None
    image_urls: list[ImageUrl] = []
