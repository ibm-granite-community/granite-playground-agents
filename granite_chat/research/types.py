from typing import Literal

from pydantic import BaseModel, Field


class ResearchEvent(BaseModel):
    event_type: Literal["log", "token"]
    data: str


class ResearchReport(BaseModel):
    topic: str
    report: str


class ResearchQuestion(BaseModel):
    question: str = Field(description="The research question.")
    search_query: str = Field(description="Optimized search query corresponding to the research question.")


class ResearchPlanSchema(BaseModel):
    research_questions: list[ResearchQuestion] = Field(
        description="A list of research questions that make up the research plan."
    )
