from typing import Literal

from pydantic import BaseModel


class ResearchEvent(BaseModel):
    event_type: Literal["log", "token"]
    data: str


class ResearchReport(BaseModel):
    task: str
    report: str
