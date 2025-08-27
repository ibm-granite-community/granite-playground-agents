from pydantic import BaseModel, Field


class ResearchQuery(BaseModel):
    query: str = Field(description="Optimized search query addressing an aspect of the research topic.")
    keywords: str = Field(description="List of keywords for this query")
    rationale: str = Field(
        description="Concise explanation of this queries place in the overall narrative, showing how it connects with previous topics to contribute to a coherent report"  # noqa: E501
    )


class ResearchReport(BaseModel):
    query: ResearchQuery
    report: str


class ResearchPlanSchema(BaseModel):
    queries: list[ResearchQuery] = Field(description="A list of search queries that address the research topic.")
