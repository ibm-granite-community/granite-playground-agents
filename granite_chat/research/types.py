from typing import Literal

from pydantic import BaseModel, Field


class ResearchEvent(BaseModel):
    event_type: Literal["log", "token"]
    data: str


class ResearchReport(BaseModel):
    topic: str
    report: str


class ResearchPlanSchema(BaseModel):
    plan: list[str] = Field(description="A list of queries/research questions that make up the research plan.")
