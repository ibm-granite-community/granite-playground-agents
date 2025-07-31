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
    # reason: str = Field(description="One line reason as to why this search results is or is not relevant.")
    is_relevant: bool = Field(description="Flag indicating if the search result is likely to be relevant.")


class StandaloneQuerySchema(BaseModel):
    query: str = Field(description="Standalone query that clearly and concisely reflects the user's intent.")


class ImageUrl(BaseModel):
    score: float
    url: str


class ScrapedContent(BaseModel):
    search_result: SearchResult
    url: str
    title: str | None = None
    raw_content: str | None = None
    image_urls: list[ImageUrl] = []
