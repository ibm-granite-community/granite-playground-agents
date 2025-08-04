from enum import Enum

from pydantic import BaseModel


class Status(str, Enum):
    active = "active"
    completed = "completed"


class Phase(BaseModel):
    status: Status
    activity: str


class SearchingWebPhase(Phase):
    activity: str = "searching-web"


class GeneratingCitationsPhase(Phase):
    activity: str = "generating-citations"
