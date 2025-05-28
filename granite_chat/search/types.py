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
