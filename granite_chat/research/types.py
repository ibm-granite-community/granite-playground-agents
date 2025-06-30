from pydantic import BaseModel, Field


class ResearchQuery(BaseModel):
    query: str = Field(description="Optimized search query addressing an aspect of the research topic.")
    keywords: str = Field(description="List of keywords for this query")


class ResearchReport(BaseModel):
    query: ResearchQuery
    report: str


class ResearchPlanSchema(BaseModel):
    queries: list[ResearchQuery] = Field(description="A list of search queries that address the research topic.")
