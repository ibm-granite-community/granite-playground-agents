from enum import Enum

from pydantic import BaseModel


class Status(str, Enum):
    active = "active"
    completed = "completed"


class Phase(BaseModel):
    status: Status
    name: str

    @property
    def wrapped(self) -> dict:
        return {"phase": self.model_dump()}


class SearchingWebPhase(Phase):
    name: str = "searching-web"


class GeneratingCitationsPhase(Phase):
    name: str = "generating-citations"
