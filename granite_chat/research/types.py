from pydantic import BaseModel, Field


class ResearchQuery(BaseModel):
    query: str = Field(description="Optimized research question addressing an aspect of the research topic.")
    keywords: str = Field(description="List of keywords for this question")
    rationale: str = Field(
        description="Provide a brief rationale explaining its importance and how it contributes to the logical flow of the investigation."  # noqa: E501
    )


class ResearchReport(BaseModel):
    query: ResearchQuery
    report: str


class ResearchPlanSchema(BaseModel):
    queries: list[ResearchQuery] = Field(description="A list of search queries that address the research topic.")
