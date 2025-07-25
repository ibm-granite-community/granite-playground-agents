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
    snippet: str

    class Config:
        frozen = True  # makes it immutable and hashable


class SearchQueriesSchema(BaseModel):
    search_queries: list[str] = Field(description="The list of search queries.")


class SearchResultRelevanceSchema(BaseModel):
    is_relevant: bool = Field(description="Flag indicating if the search result is likely to be relevant.")


class ImageUrl(BaseModel):
    score: float
    url: str


class ScrapedContent(BaseModel):
    search_result: SearchResult
    url: str
    title: str | None = None
    raw_content: str | None = None
    image_urls: list[ImageUrl] = []
