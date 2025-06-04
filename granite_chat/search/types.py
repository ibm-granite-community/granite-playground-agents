from pydantic import BaseModel


class SearchResult(BaseModel):
    title: str
    href: str
    body: str

    @property
    def url(self) -> str:
        return self.href


class SearchResults(BaseModel):
    results: list[SearchResult] = []


class Source(BaseModel):
    url: str
    title: str

    class Config:
        frozen = True  # makes it immutable and hashable


class Citation(BaseModel):
    citation_id: str
    source: str
    source_title: str
    context_text: str
    context_begin: int
    context_end: int
    response_text: str
    response_begin: int
    response_end: int
