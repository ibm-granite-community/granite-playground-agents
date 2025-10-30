# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0


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
    # rationale: str = Field(description="Brief one sentence rationale explaining your decision.")
    is_relevant: bool = Field(description="Flag indicating if the search result is likely to be relevant.")


class StandaloneQuerySchema(BaseModel):
    query: str = Field(description="Standalone query that clearly and concisely reflects the user's intent.")
