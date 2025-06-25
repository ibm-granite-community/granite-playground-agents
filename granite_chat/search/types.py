from pydantic import BaseModel, Field


class SearchResult(BaseModel):
    title: str
    href: str
    body: str

    @property
    def url(self) -> str:
        return self.href


class Source(BaseModel):
    url: str
    title: str

    class Config:
        frozen = True  # makes it immutable and hashable


class Citation(BaseModel):
    citation_id: str
    document_id: str
    source: str
    source_title: str
    context_text: str
    context_begin: int
    context_end: int
    response_text: str
    response_begin: int
    response_end: int


class SearchQueriesSchema(BaseModel):
    search_queries: list[str] = Field(description="The list of search queries.")


class ImageUrl(BaseModel):
    score: float
    url: str


class ScrapedContent(BaseModel):
    search_result: SearchResult
    url: str
    title: str | None = None
    raw_content: str | None = None
    image_urls: list[ImageUrl] = []
